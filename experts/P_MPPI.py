# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:09:35 2024

@author: siliconsynapse
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import math
from jax import config
config.update("jax_enable_x64", True)

from sklearn.covariance import OAS

from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,JU_RANGE_SFIM,Single_FIM_Radar,FIM_Visualization
from src_range.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range.utils import visualize_tracking,visualize_tracking_NN,visualize_control,visualize_target_mse,place_sensors_restricted
from src_range.control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper #,MPPI_adapt_distribution
from src_range.objective_fns.objectives import *
from src_range.tracking.cubatureTestMLP import generate_data_state


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle

import imageio

from time import time
import os
import argparse
from copy import deepcopy
import shutil

class P_MPPI:
    def __init__(self, state_shape, n_actions):
        

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.MPPI_iterations=10
        self.gamma=0.95
        self.temperature=0.1
        self.num_traj=500
        self.seed = 123
        self.v_init = 0
        self.av_init = 0
        
    def predict_probs(self,mean,cov,x):
       x=x.T
       mean=mean.T
       pdf=(2*math.pi)**(-1)*(jnp.linalg.det(cov))**(-1/2)*jnp.exp(-1/2*jnp.matmul(jnp.matmul((x-mean).T,jnp.linalg.inv(cov)),(x-mean)))
       return pdf
        
    def generate_session(self, args,epochs,state_train,D_demo,mpc_method=None,thetas=None):
        
        
        demo_x_min,demo_x_max= jnp.min(D_demo[:,0]),jnp.max(D_demo[:,0])
        demo_y_min,demo_y_max= jnp.min(D_demo[:,1]),jnp.max(D_demo[:,1])
        states, traj_probs, actions,FIMs = [], [], [],[]
        
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mplstyle.use('fast')
        mplstyle.use(['ggplot', 'fast'])
        key = jax.random.PRNGKey(args.seed)
        np.random.seed(args.seed)
        traj_sindy_list=[]

        # =========================== Experiment Choice ================== #
        update_freq_control = int(args.dt_control/args.dt_ckf) #4
        update_freq_ckf = 1

        # ==================== RADAR setup ======================== #
        # speed of light
        c = 299792458
        K = args.Pt * args.Gt * args.Gr * (c/args.fc) ** 2 * args.rcs / args.L / (4 * jnp.pi) ** 3
        Pr = K / (args.R ** 4)

        # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
        control_constraints = UNI_DI_U_LIM
        kinematic_model = unicycle_kinematics_double_integrator

        # ==================== MPPI CONFIGURATION ================================= #
        cov_timestep = jnp.array([[args.acc_std**2,0],[0,args.ang_acc_std**2]])
        cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])
        cov = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(args.N_radar)])
        # cov = jnp.stack([cov_traj for n in range(N)])

        #mpc_method = "Single_FIM_3D_action_features_MPPI"

        # ==================== AIS CONFIGURATION ================================= #
        key, subkey = jax.random.split(key)
        #

        z_elevation = 60
        # target_state = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
        #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
        #                 # [10,10,z_elevation,10,10,0],
        #                 [20,20,z_elevation,5,-5,0]])
        target_state = jnp.array([[0.0, -0.0,z_elevation-5, 20., 10,0]])
        # target_state = jnp.array([[0.0, -0.0,z_elevation+10, 25., 20,0], #,#,
        #                 [-100.4,-30.32,z_elevation-15,20,-10,0], #,
        #                 [30,30,z_elevation+20,-10,-10,0]])#,

        ps,key = place_sensors_restricted(key,target_state,args.R2R,args.R2T,-400,400,args.N_radar)
        ps=D_demo[0,:3].reshape((1,3))
        chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi)
        vs = jnp.zeros((ps.shape[0],1))
        avs = jnp.zeros((ps.shape[0],1))
        radar_state = jnp.column_stack((ps,chis,vs,avs))
        radar_state_init = deepcopy(radar_state)


        M_target, dm = target_state.shape;
        _ , dn = radar_state.shape;

        sigmaW = jnp.sqrt(M_target*Pr/ (10**(args.SNR/10)))
        # coef = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi)** 3 / (R ** 4)
        C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * args.fc**2) * 1/K

        print("Noise Power: ",sigmaW**2)
        print("Power Return (RCS): ",Pr)
        print("K",K)

        print("Pt (peak power)={:.9f}".format(args.Pt))
        print("lam ={:.9f}".format(c/args.fc))
        print("C=",C)

        # ========================= Target State Space ============================== #

        sigmaQ = np.sqrt(10 ** 1)

        A_single = jnp.array([[1., 0, 0, args.dt_control, 0, 0],
                       [0, 1., 0, 0, args.dt_control, 0],
                       [0, 0, 1, 0, 0, args.dt_control],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1., 0],
                       [0, 0, 0, 0, 0, 1]])

        Q_single = jnp.array([
            [(args.dt_control ** 4) / 4, 0, 0, (args.dt_control ** 3) / 2, 0, 0],
            [0, (args.dt_control ** 4) / 4, 0, 0, (args.dt_control** 3) / 2, 0],
            [0, 0, (args.dt_control**4)/4, 0, 0, (args.dt_control**3) / 2],
            [(args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0, 0],
            [0, (args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0],
            [0, 0, (args.dt_control**3) / 2, 0, 0, (args.dt_control**2)]
        ]) * sigmaQ ** 2

        A = jnp.kron(jnp.eye(M_target), A_single);
        Q = jnp.kron(jnp.eye(M_target), Q_single);
        # Q = Q + jnp.eye(Q.shape[0])*1e-6
        #
        nx = Q.shape[0]

       

        if args.fim_method_policy == "PFIM":
            IM_fn = partial(Single_JU_FIM_Radar, A=A, Q=Q, C=C)
            IM_fn_update = partial(Single_JU_FIM_Radar, A=A_ckf, Q=Q_ckf, C=C)
        elif args.fim_method_policy == "SFIM":
            IM_fn = partial(Single_FIM_Radar, C=C)
            IM_fn_update = IM_fn
        elif args.fim_method_policy == "SFIM_NN" or args.fim_method_policy == "SFIM_features" :
            IM_fn = partial(Single_FIM_Radar, C=C)
            IM_fn_update = IM_fn
        elif args.fim_method_policy == "SFIM_bad":
            sigmaR = 1
            IM_fn = partial(JU_RANGE_SFIM, R=jnp.eye(M_target*args.N_radar)*sigmaR**2)
            IM_fn_update = IM_fn


        MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method,state_train=state_train,thetas=thetas,gail=args.gail)
        #MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method)
        MPPI_scores = MPPI_scores_wrapper(MPC_obj)

        MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)

        if args.AIS_method == "CE":
            weight_fn = partial(weighting(args.AIS_method),elite_threshold=args.elite_threshold)
        elif args.AIS_method == "information":
            weight_fn = partial(weighting(args.AIS_method),temperature=args.temperature)

        # weight_info =partial(weighting("information"),temperature=temperature)

        chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
        # dt_controls = jnp.tile(dt_control, (N, 1))

        collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None,None,None,None)))
        self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None)))
        speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))


        U1 = jnp.ones((args.N_radar,args.horizon,1)) #* args.acc_init
        U2 = jnp.ones((args.N_radar,args.horizon,1)) #* args.ang_acc_init
        U =jnp.concatenate((U1,U2),axis=-1)

        if not args.move_radars:
            U = jnp.zeros_like(U)
            radar_states_MPPI = None
            cost_MPPI = None

        # # generate radar states at measurement frequency
        # radar_states = kinematic_model(np.repeat(U, update_freq_control+1, axis=1)[:, :update_freq_control,:],
        #                                radar_state, args.dt_ckf)

        # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,args.horizon,2),axis=0),U_lower,U_upper)

        # generate the true target state
        target_states_true = jnp.array(generate_data_state(target_state,args.N_steps, M_target, dm,dt=args.dt_control,Q=Q))

        # radar state history
        radar_state_history = np.zeros((args.N_steps+1,)+radar_state.shape)

        FIMs = np.zeros(args.N_steps//update_freq_control + 1)
        FIMs_NN = np.zeros(args.N_steps//update_freq_control + 1)

        fig_main,axes_main = plt.subplots(1,2,figsize=(10,5))
        imgs_main =  []

        fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
        imgs_control =  []


        fig_mse,axes_mse = plt.subplots(1,figsize=(10,5))
        target_state_mse = np.zeros(args.N_steps)
        P=np.eye(M_target*dm) * 50
        
        J_NN = jnp.linalg.inv(P)
        J=J_NN.copy()
        
        
        
        for step in range(1,args.N_steps+1):
            target_state_true = target_states_true[:, step-1].reshape(M_target,dm)
            
            
            best_mppi_iter_score = np.inf
            mppi_round_time_start = time()

            # need dimension Horizon x Number of Targets x Dim of Targets
            # target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,horizon+1)])

            if args.move_radars:
                #(4, 15, 6)
                # the cubature kalman filter points propogated over horizon. Horizon x # Sigma Points (2*dm) x (Number of targets * dim of target)
                target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ target_state_true.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,args.horizon+1)])
                target_states_rollout = np.swapaxes(target_states_rollout, 1, 0)

                mppi_start_time = time()

                U_prime = deepcopy(U)
                cov_prime = deepcopy(cov)
            

                print(f"\n Step {step} MPPI CONTROL ")


                for mppi_iter in range(args.MPPI_iterations):
                    start = time()
                    key, subkey = jax.random.split(key)

                    mppi_start = time()

                    E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(args.num_traj,),method="svd")

                    # simulate the model with the trajectory noise samples
                    V = U_prime + E.reshape(args.num_traj,args.N_radar,args.horizon,2)

                    mppi_rollout_start = time()

                    radar_states,radar_states_MPPI = MPPI(U_nominal=U_prime,
                                                                       U_MPPI=V,radar_state=radar_state)
                    
                    # radar_states=radar_states.at[:,:,0].set(jnp.clip(radar_states[:,:,0],demo_x_min,demo_x_max))
                    # radar_states=radar_states.at[:,:,1].set(jnp.clip(radar_states[:,:,1],demo_y_min,demo_y_max))
                    # radar_states_MPPI=radar_states_MPPI.at[:,:,:,0].set(jnp.clip(radar_states_MPPI[:,:,:,0],demo_x_min,demo_x_max))
                    # radar_states_MPPI=radar_states_MPPI.at[:,:,:,1].set(jnp.clip(radar_states_MPPI[:,:,:,1],demo_y_min,demo_y_max))
                    

                    mppi_rollout_end = time()
                    

                    # GET MPC OBJECTIVE
                    mppi_score_start = time()
                    # Score all the rollouts
                    cost_trajectory = MPPI_scores(radar_state, target_states_rollout, V,
                                              A=A,J=J)
                    

                    mppi_score_end = time()
                    cost_collision_r2t = collision_penalty_vmap(radar_states_MPPI[...,1:args.horizon+1,:], target_states_rollout,
                                           demo_x_min,demo_x_max,demo_y_min,demo_y_max)

                    cost_collision_r2t = jnp.sum((cost_collision_r2t * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                    
                    #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                    cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t

                    weights = weight_fn(cost_MPPI)


                    if jnp.isnan(cost_MPPI).any():
                        print("BREAK!")
                        break

                    if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):

                        best_cost = jnp.sum(cost_MPPI*weights)

                        U_copy = deepcopy(U_prime)
                        U_prime = U_prime + jnp.sum(weights.reshape(args.num_traj,1,1,1) * E.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

                        oas = OAS(assume_centered=True).fit(E[weights != 0])
                        cov_prime = jnp.array(oas.covariance_)
                        if mppi_iter == 0 and epochs%10 ==0:
                            print("Oracle Approx Shrinkage: ",np.round(oas.shrinkage_,5))

                mppi_round_time_end = time()

                if jnp.isnan(cost_MPPI).any():
                    print("BREAK!")
                    break

                weights = weight_fn(cost_MPPI)

                mean_shift = (U_prime - U)

                E_prime = E + mean_shift.ravel()

                U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

                U = jnp.stack((jnp.clip(U[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)

                # jnp.repeat(U,update_freq_control,axis=1)

                # radar_states = kinematic_model(U ,radar_state, dt_control)

                # generate radar states at measurement frequency
                #U=U.at[:,0,:].set(actions_d[step-1,:].reshape((1,2)))
                #U[:,0,:]=actions_d[step,:].reshape((1,2))
                radar_states = kinematic_model(U,
                                               radar_state, args.dt_control)
                

                #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)

                radar_state = radar_states[:,1]
                # radar_state=radar_state.at[:,0].set(jnp.clip(radar_state[:,0],demo_x_min,demo_x_max))
                # radar_states=radar_state.at[:,1].set(jnp.clip(radar_state[:,1],demo_y_min,demo_y_max))
                
                U = jnp.roll(U, -1, axis=1)

                mppi_end_time = time()
                
                print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))



            
            
            
            
            
           
            probs_fun=vmap(self.predict_probs,(0,None,0))
            prob=self.predict_probs(U[:,0,:], cov_prime[:2,:2],U[:,0,:])
            prob=jnp.array([1])


            state=jnp.concatenate((radar_state[:,:3].flatten(),target_state_true[:,0:3].flatten()))
            
            states.append(state)
            traj_probs.append(prob.flatten())
            actions.append(U[:,0,:].flatten())
            

            J_NN = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                      J=J,method="NN",state_train=state_train,thetas=thetas)
            J = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                      J=J)
            
            traj_sindy=jnp.concatenate((radar_state[:,:3],target_state_true[:,0:3]))
            traj_sindy_list.append(traj_sindy)
            # print(jnp.linalg.slogdet(J)[1].ravel().item())
            if args.N_radar==1 and M_target==1:
                FIMs_NN[step] = jnp.log(J_NN).ravel().item()
                FIMs[step] = jnp.log(J).ravel().item()
            else:
                FIMs_NN[step] = jnp.linalg.slogdet(J_NN)[1].ravel().item()
                FIMs[step] = jnp.linalg.slogdet(J)[1].ravel().item()

            radar_state_history[step] = radar_state
           
            print("FIM :" , FIMs[step])
            print("Thetas :" , thetas)
            
            
            if args.save_images and (step % 4 == 0):
                print(f"Step {step} - Saving Figure ")

                axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                         label="Radar Initial Position")

                thetas_1 = jnp.arcsin(target_state_true[:, 2] / args.R2T)
                radius_projected = args.R2T * jnp.cos(thetas_1)

                print("Target Height :",target_state_true[:,2])
                print("Radius Projected: ",radius_projected)

                # radar_state_history[step] = radar_state

                try:
                    imgs_main.append(visualize_tracking_NN(target_state_true=target_state_true, target_state_ckf=target_state_true,target_states_true=target_states_true.T.reshape(-1,M_target,dm)[:step],
                               radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,radar_state_history=radar_state_history[max(step // update_freq_control - args.tail_length,0):step // update_freq_control],
                               cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                               R2T=args.R2T, R2R=args.R2R,C=C,
                               fig=fig_main, axes=axes_main, step=step,
                               tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_NN",state_train=state_train))
                except Exception as error:
                    print("Tracking Img Could Not save: ",error)

               
                try:
                    imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                               fig=fig_control, axes=axes_control, step=step,
                               tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_control_NN"))
                except:
                    print("Control Img Could Not Save")


            # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)

            # CKF ! ! ! !
           

        
        np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')

        if args.save_images:
            visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

            images = [imageio.imread(file) for file in imgs_main]
            imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method_policy}.gif'), images, duration=0.1)

            images = [imageio.imread(file) for file in imgs_control]
            imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


            if args.remove_tmp_images:
                shutil.rmtree(args.tmp_img_savepath)
    
           
               
        return states,traj_probs,actions,FIMs,FIMs_NN
    
    
    
    def generate_session_N_CIRL(self, args,epochs,target_states,state_train,D_demo,mpc_method=None,thetas=None):

         demo_x_min,demo_x_max= jnp.min(D_demo[:,0]),jnp.max(D_demo[:,0])
         demo_y_min,demo_y_max= jnp.min(D_demo[:,1]),jnp.max(D_demo[:,1])
         states, traj_probs, actions,FIMs = [], [], [],[]
         
         
         mpl.rcParams['path.simplify_threshold'] = 1.0
         mplstyle.use('fast')
         mplstyle.use(['ggplot', 'fast'])
         key = jax.random.PRNGKey(args.seed)
         key_t = jax.random.PRNGKey(args.seed)
         np.random.seed(args.seed)
    
         # =========================== Experiment Choice ================== #
         update_freq_control = int(args.dt_control/args.dt_ckf) #4
         update_freq_ckf = 1
         traj_list=[]
         traj_sindy_list=[]
    
         # ==================== RADAR setup ======================== #
         # speed of light
         c = 299792458
         K = args.Pt * args.Gt * args.Gr * (c/args.fc) ** 2 * args.rcs / args.L / (4 * jnp.pi) ** 3
         Pr = K / (args.R ** 4)
    
         # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
         control_constraints = UNI_DI_U_LIM
         kinematic_model = unicycle_kinematics_double_integrator
    
         # ==================== MPPI CONFIGURATION ================================= #
         cov_timestep = jnp.array([[args.acc_std**2,0],[0,args.ang_acc_std**2]])
         cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])
         cov = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(args.N_radar)])
         cov_t = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(args.N_radar)])
         # cov = jnp.stack([cov_traj for n in range(N)])
    
         mpc_method = "Single_FIM_3D_action_NN_MPPI"
        
         mpc_method_t = "Single_FIM_3D_evasion_NN_MPPI"
    
         # ==================== AIS CONFIGURATION ================================= #
         key, subkey = jax.random.split(key)
         #
    
         z_elevation = 60
         # target_state = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
         #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
         #                 # [10,10,z_elevation,10,10,0],
         #                 [20,20,z_elevation,5,-5,0]])
         target_state_v = jnp.array([[0.0, -0.0,z_elevation-5, 20., 10,0]])
         # target_state = jnp.array([[0.0, -0.0,z_elevation+10, 25., 20,0], #,#,
         #                 [-100.4,-30.32,z_elevation-15,20,-10,0], #,
         #                 [30,30,z_elevation+20,-10,-10,0]])#,
    
         ps,key = place_sensors_restricted(key,target_state_v,args.R2R,args.R2T,-400,400,args.N_radar)
         chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi)
         vs = jnp.zeros((ps.shape[0],1))
         avs = jnp.zeros((ps.shape[0],1))
         radar_state = jnp.column_stack((ps,chis,vs,avs))
         radar_state_init = deepcopy(radar_state)
         pt=target_states[0,:3].reshape(1,-1)
         chit = jax.random.uniform(key,shape=(pt.shape[0],1),minval=-jnp.pi,maxval=jnp.pi)
         target_state=jnp.column_stack((pt,chit,vs,avs))
    
    
         M_target, dm = target_state_v.shape;
         _ , dn = radar_state.shape;
    
         sigmaW = jnp.sqrt(M_target*Pr/ (10**(args.SNR/10)))
         # coef = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi)** 3 / (R ** 4)
         C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * args.fc**2) * 1/K
    
         print("Noise Power: ",sigmaW**2)
         print("Power Return (RCS): ",Pr)
         print("K",K)
    
         print("Pt (peak power)={:.9f}".format(args.Pt))
         print("lam ={:.9f}".format(c/args.fc))
         print("C=",C)
    
         # ========================= Target State Space ============================== #
    
         sigmaQ = np.sqrt(10 ** 1)
    
         A_single = jnp.array([[1., 0, 0, args.dt_control, 0, 0],
                        [0, 1., 0, 0, args.dt_control, 0],
                        [0, 0, 1, 0, 0, args.dt_control],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1., 0],
                        [0, 0, 0, 0, 0, 1]])
    
         Q_single = jnp.array([
             [(args.dt_control ** 4) / 4, 0, 0, (args.dt_control ** 3) / 2, 0, 0],
             [0, (args.dt_control ** 4) / 4, 0, 0, (args.dt_control** 3) / 2, 0],
             [0, 0, (args.dt_control**4)/4, 0, 0, (args.dt_control**3) / 2],
             [(args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0, 0],
             [0, (args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0],
             [0, 0, (args.dt_control**3) / 2, 0, 0, (args.dt_control**2)]
         ]) * sigmaQ ** 2
    
         A = jnp.kron(jnp.eye(M_target), A_single);
         Q = jnp.kron(jnp.eye(M_target), Q_single);
         # Q = Q + jnp.eye(Q.shape[0])*1e-6
         #
         nx = Q.shape[0]
    
        
    
         
         if args.fim_method_policy == "PFIM":
              IM_fn = partial(Single_JU_FIM_Radar, A=A, Q=Q, C=C)
              IM_fn_update = partial(Single_JU_FIM_Radar, A=A_ckf, Q=Q_ckf, C=C)
         elif args.fim_method_policy == "SFIM":
              IM_fn = partial(Single_FIM_Radar, C=C)
              IM_fn_update = IM_fn
         elif args.fim_method_policy == "SFIM_NN" or args.fim_method_policy == "SFIM_features" :
              IM_fn = partial(Single_FIM_Radar, C=C)
              IM_fn_update = IM_fn
         elif args.fim_method_policy == "SFIM_bad":
              sigmaR = 1
              IM_fn = partial(JU_RANGE_SFIM, R=jnp.eye(M_target*args.N_radar)*sigmaR**2)
              IM_fn_update = IM_fn
          
         
    
         MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method,state_train=state_train,thetas=thetas)
    
         MPPI_scores = MPPI_scores_wrapper(MPC_obj)
    
         MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)
         
         
         MPC_obj_t = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method_t,state_train=state_train)
    
         MPPI_scores_t = MPPI_scores_wrapper(MPC_obj_t)
    
         MPPI_t= MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)
         
         
    
    
         if args.AIS_method == "CE":
             weight_fn = partial(weighting(args.AIS_method),elite_threshold=args.elite_threshold)
         elif args.AIS_method == "information":
             weight_fn = partial(weighting(args.AIS_method),temperature=args.temperature)
    
         # weight_info =partial(weighting("information"),temperature=temperature)
    
         chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
         # dt_controls = jnp.tile(dt_control, (N, 1))
    
         collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None,None,None,None)))
         self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None)))
         speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))
    
    
         U1 = jnp.ones((args.N_radar,args.horizon,1)) #* args.acc_init
         U2 = jnp.ones((args.N_radar,args.horizon,1)) #* args.ang_acc_init
         U =jnp.concatenate((U1,U2),axis=-1)
         U_t =jnp.concatenate((U1,U2),axis=-1)
      
    
         if not args.move_radars:
             U = jnp.zeros_like(U)
             U_t = jnp.zeros_like(U_t)
             
             radar_states_MPPI = None
             cost_MPPI = None
    
         # # generate radar states at measurement frequency
         # radar_states = kinematic_model(np.repeat(U, update_freq_control+1, axis=1)[:, :update_freq_control,:],
         #                                radar_state, args.dt_ckf)
    
         # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,args.horizon,2),axis=0),U_lower,U_upper)
    
         # generate the true target state
        # target_states_true = jnp.array(generate_data_state(target_state,args.N_steps, M_target, dm,dt=args.dt_control,Q=Q))
    
         # radar state history
         radar_state_history = np.zeros((args.N_steps+1,)+radar_state.shape)
    
         FIMs = np.zeros(args.N_steps//update_freq_control + 1)
    
    
         fig_main,axes_main = plt.subplots(1,2,figsize=(10,5))
         imgs_main =  []
    
         fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
         imgs_control =  []
    
    
         fig_mse,axes_mse = plt.subplots(1,figsize=(10,5))
         target_state_mse = np.zeros(args.N_steps)
         P=np.eye(M_target*dm) * 50
         
         J = jnp.linalg.inv(P)
    
         for step in range(1,args.N_steps+1):
             #target_state_true = target_states_true[:, step-1].reshape(M_target,dm)
             target_state_v=jnp.concatenate((pt,target_state_v[:,3:]),axis=1)
             
    
    
    
    
             best_mppi_iter_score = np.inf
             mppi_round_time_start = time()
    
             # need dimension Horizon x Number of Targets x Dim of Targets
             # target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,horizon+1)])
    
             if args.move_radars:
                 #(4, 15, 6)
                 # the cubature kalman filter points propogated over horizon. Horizon x # Sigma Points (2*dm) x (Number of targets * dim of target)
                 target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ target_state_v.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,args.horizon+1)])
                 target_states_rollout = np.swapaxes(target_states_rollout, 1, 0)
    
                 mppi_start_time = time()
    
                 U_prime = deepcopy(U)
                 cov_prime = deepcopy(cov)
    
                 print(f"\n Step {step} MPPI CONTROL ")
    
    
                 for mppi_iter in range(args.MPPI_iterations):
                     start = time()
                     key, subkey = jax.random.split(key)
    
                     mppi_start = time()
    
                     E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(args.num_traj,),method="svd")
    
                     # simulate the model with the trajectory noise samples
                     V = U_prime + E.reshape(args.num_traj,args.N_radar,args.horizon,2)
    
                     mppi_rollout_start = time()
    
                     radar_states,radar_states_MPPI = MPPI(U_nominal=U_prime,
                                                                        U_MPPI=V,radar_state=radar_state)
                     # radar_states=radar_states.at[:,:,0].set(jnp.clip(radar_states[:,:,0],demo_x_min,demo_x_max))
                     # radar_states=radar_states.at[:,:,1].set(jnp.clip(radar_states[:,:,1],demo_y_min,demo_y_max))
                     # radar_states_MPPI=radar_states_MPPI.at[:,:,:,0].set(jnp.clip(radar_states_MPPI[:,:,:,0],demo_x_min,demo_x_max))
                     # radar_states_MPPI=radar_states_MPPI.at[:,:,:,1].set(jnp.clip(radar_states_MPPI[:,:,:,1],demo_y_min,demo_y_max))
                     
    
                     mppi_rollout_end = time()
                     
    
                     # GET MPC OBJECTIVE
                     mppi_score_start = time()
                     # Score all the rollouts
                     cost_trajectory = MPPI_scores(radar_state, target_states_rollout, V,
                                               A=A,J=J)
                     
    
                     mppi_score_end = time()
                     
                     cost_collision_r2t = collision_penalty_vmap(radar_states_MPPI[...,1:args.horizon+1,:], target_states_rollout,
                                             args.x_min,args.x_max,args.y_min,args.y_max)

                     cost_collision_r2t = jnp.sum((cost_collision_r2t * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                     #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                     cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t
    
                     weights = weight_fn(cost_MPPI)
    
    
                     if jnp.isnan(cost_MPPI).any():
                         print("BREAK!")
                         break
    
                     if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):
    
                         best_cost = jnp.sum(cost_MPPI*weights)
    
                         U_copy = deepcopy(U_prime)
                         U_prime = U_prime + jnp.sum(weights.reshape(args.num_traj,1,1,1) * E.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
    
                         oas = OAS(assume_centered=True).fit(E[weights != 0])
                         cov_prime = jnp.array(oas.covariance_)
                         if mppi_iter == 0:
                             print("Oracle Approx Shrinkage: ",np.round(oas.shrinkage_,5))
    
                 mppi_round_time_end = time()
    
                 if jnp.isnan(cost_MPPI).any():
                     print("BREAK!")
                     break
    
                 weights = weight_fn(cost_MPPI)
    
                 mean_shift = (U_prime - U)
    
                 E_prime = E + mean_shift.ravel()
    
                 U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
    
                 U = jnp.stack((jnp.clip(U[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)
    
    
          
    
                # generate radar states at measurement frequency
                 radar_states = kinematic_model(U,
                                               radar_state, args.dt_control)
    
                #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)
    
                 radar_state = radar_states[:,1]
                 # radar_state=radar_state.at[:,0].set(jnp.clip(radar_state[:,0],demo_x_min,demo_x_max))
                 # radar_state=radar_state.at[:,1].set(jnp.clip(radar_state[:,1],demo_y_min,demo_y_max))
                 U = jnp.roll(U, -1, axis=1)
                 ps=radar_state[:,:3]
    
                
                
    
                #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)
    
    
                 mppi_end_time = time()
                 print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))
                
                  
                
                ## target cost function movement
                 # U_prime_t = deepcopy(U_t)
                 # cov_prime_t = deepcopy(cov_t)
         
                 # print(f"\n Step {step} MPPI CONTROL TARGET ")
                 
                 
                 # for mppi_iter in range(args.MPPI_iterations):
                 #     start = time()
                 #     key_t, subkey = jax.random.split(key_t)
         
                 #     mppi_start = time()
         
                 #     E_t = jax.random.multivariate_normal(key_t, mean=jnp.zeros_like(U_t).ravel(), cov=cov_prime_t, shape=(args.num_traj,),method="svd")
         
                 #     # simulate the model with the trajectory noise samples
                 #     V_t = U_prime_t + E_t.reshape(args.num_traj,args.N_radar,args.horizon,2)
         
                 #     mppi_rollout_start = time()
         
                 #     target_states,target_states_MPPI = MPPI(U_nominal=U_prime_t,
                 #                                                        U_MPPI=V_t,radar_state=target_state)
                 #     # target_states=target_states.at[:,:,0].set(jnp.clip(target_states[:,:,0],demo_x_min,demo_x_max))
                 #     # target_states=target_states.at[:,:,1].set(jnp.clip(target_states[:,:,1],demo_y_min,demo_y_max))
                 #     # target_states_MPPI=target_states_MPPI.at[:,:,:,0].set(jnp.clip(target_states_MPPI[:,:,:,0],demo_x_min,demo_x_max))
                 #     # target_states_MPPI=target_states_MPPI.at[:,:,:,1].set(jnp.clip(target_states_MPPI[:,:,:,1],demo_y_min,demo_y_max))
                     
         
                 #     mppi_rollout_end = time()
         
         
                 #     # GET MPC OBJECTIVE
                 #     mppi_score_start = time()
                 #     # Score all the rollouts
                 #     cost_trajectory_t = MPPI_scores_t(radar_states, target_state, V_t,
                 #                               A=A,J=J)
         
                 #     mppi_score_end = time()
         
         
                     
                 #     #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                 #     cost_MPPI_t = args.alpha1*cost_trajectory_t
         
                 #     weights_t = weight_fn(cost_MPPI_t)
         
         
                 #     if jnp.isnan(cost_MPPI_t).any():
                 #         print("BREAK!")
                 #         break
         
                 #     if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):
         
                 #         best_cost_t = jnp.sum(cost_MPPI_t*weights_t)
         
                 #         U_copy_t = deepcopy(U_prime_t)
                 #         U_prime_t = U_prime_t + jnp.sum(weights_t.reshape(args.num_traj,1,1,1) * E_t.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
         
                 #         oas_t = OAS(assume_centered=True).fit(E_t[weights_t != 0])
                 #         cov_prime_t = jnp.array(oas_t.covariance_)
                 #         if mppi_iter == 0:
                 #             print("Oracle Approx Shrinkage: ",np.round(oas_t.shrinkage_,5))
                 
                 # mppi_round_time_end = time()
         
                 # if jnp.isnan(cost_MPPI_t).any():
                 #     print("BREAK!")
                 #     break
         
                 # weights_t = weight_fn(cost_MPPI_t)
         
                 # mean_shift_t = (U_prime_t - U_t)
         
                 # E_prime_t = E_t + mean_shift_t.ravel()
         
                 # U_t += jnp.sum(weights_t.reshape(-1,1,1,1) * E_prime_t.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
         
                 # U_t = jnp.stack((jnp.clip(U_t[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U_t[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)
         
                 # # jnp.repeat(U,update_freq_control,axis=1)
         
                 # target_states = kinematic_model(U_t ,target_state, args.dt_control)
         
                 # # generate radar states at measurement frequency
                 # #target_states = kinematic_model(np.repeat(U_t, update_freq_control, axis=1)[:, :update_freq_control, :],
                 # #                              target_state, args.dt_control)
         
                 # # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)
         
                 # target_state = target_states[:,1]
                 # # target_state=target_state.at[:,0].set(jnp.clip(target_state[:,0],demo_x_min,demo_x_max))
                 # # target_state=target_state.at[:,1].set(jnp.clip(target_state[:,1],demo_y_min,demo_y_max))
                 # 
                 # U_t = jnp.roll(U_t, -1, axis=1)
         
                 # mppi_end_time = time()
                 # print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))     
                 
             probs_fun=vmap(self.predict_probs,(0,None,0))
             prob=self.predict_probs(U[:,0,:], cov_prime[:2,:2],U[:,0,:])
             prob=jnp.array([1])
    
                
             pt=target_states[step,:3].reshape(1,-1)
             state=jnp.concatenate((radar_state[:,:3].flatten(),target_states[step,0:3].flatten()))
             states.append(state)
             traj_probs.append(prob.flatten())
             actions.append(U[:,0,:].flatten())
             J = IM_fn_update(radar_state=radar_state, target_state=target_state,
                       J=J,method="NN",state_train=state_train,thetas=thetas)
             
             traj=np.concatenate((ps.flatten(),target_state[:,:3].flatten(),U[:,0,:].flatten()),axis=0)
             traj_sindy=np.concatenate((ps,target_state[:,:3]),axis=0)
             traj_list.append(traj)
             traj_sindy_list.append(traj_sindy)
             # print(jnp.linalg.slogdet(J)[1].ravel().item())
             if args.N_radar==1 and M_target==1:
                 FIMs[step] = jnp.log(J).ravel().item()
             else:
                 
                 FIMs[step] = jnp.linalg.slogdet(J)[1].ravel().item()
    
             radar_state_history[step] = radar_state
             print("FIM :" , FIMs[step])
    
             if args.save_images and (step % 4 == 0):
                 print(f"Step {step} - Saving Figure ")
    
                 axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                          label="Radar Initial Position")
    
                 thetas = jnp.arcsin(target_state_v[:, 2] / args.R2T)
                 radius_projected = args.R2T * jnp.cos(thetas)
    
                 print("Target Height :",target_state_v[:,2])
                 print("Radius Projected: ",radius_projected)
    
                 # radar_state_history[step] = radar_state
                 

            
    
                 try:
                     imgs_main.append(visualize_tracking_NN(target_state_true=target_state_v, target_state_ckf=target_state_v,target_states_true=None,
                                radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,radar_state_history=radar_state_history[max(step // update_freq_control - args.tail_length,0):step // update_freq_control],
                                cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                                R2T=args.R2T, R2R=args.R2R,C=C,
                                fig=fig_main, axes=axes_main, step=step,
                                tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_NN",state_train=state_train))
                 except Exception as error:
                     print("Tracking Img Could Not save: ",error)
    
                
                 try:
                     imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                                fig=fig_control, axes=axes_control, step=step,
                                tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_control_NN"))
                 except:
                     print("Control Img Could Not Save")
    
    
             # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)
    
             # CKF ! ! ! !
            
    
    
         np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')
    
         if args.save_images:
             visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

             images = [imageio.imread(file) for file in imgs_main]
             imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method_policy}.gif'), images, duration=0.1)

             images = [imageio.imread(file) for file in imgs_control]
             imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


             if args.remove_tmp_images:
                 shutil.rmtree(args.tmp_img_savepath)
            
         return states,traj_probs,actions
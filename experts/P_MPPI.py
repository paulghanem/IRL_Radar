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
        
    def generate_session(self, args,epochs,actions_d,state_train,D_demo,mpc_method=None,thetas=None):
        
        
        demo_x_min,demo_x_max= jnp.min(D_demo[:,0]),jnp.max(D_demo[:,0])
        demo_y_min,demo_y_max= jnp.min(D_demo[:,1]),jnp.max(D_demo[:,1])
        states, traj_probs, actions,FIMs = [], [], [],[]
        
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mplstyle.use('fast')
        mplstyle.use(['ggplot', 'fast'])
        key = jax.random.PRNGKey(args.seed)
        np.random.seed(args.seed)

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
        elif args.fim_method_policy == "SFIM_NN" :
            IM_fn = partial(Single_FIM_Radar, C=C)
            IM_fn_update = IM_fn
        elif args.fim_method_policy == "SFIM_bad":
            sigmaR = 1
            IM_fn = partial(JU_RANGE_SFIM, R=jnp.eye(M_target*args.N_radar)*sigmaR**2)
            IM_fn_update = IM_fn


        MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method,state_train=state_train,thetas=thetas)
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

        collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None)))
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


        fig_main,axes_main = plt.subplots(1,2,figsize=(10,5))
        imgs_main =  []

        fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
        imgs_control =  []


        fig_mse,axes_mse = plt.subplots(1,figsize=(10,5))
        target_state_mse = np.zeros(args.N_steps)
        P=np.eye(M_target*dm) * 50
        
        J = jnp.linalg.inv(P)
        
        
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
                    
                    radar_states=radar_states.at[:,:,0].set(jnp.clip(radar_states[:,:,0],demo_x_min,demo_x_max))
                    radar_states=radar_states.at[:,:,1].set(jnp.clip(radar_states[:,:,1],demo_y_min,demo_y_max))
                    radar_states_MPPI=radar_states_MPPI.at[:,:,:,0].set(jnp.clip(radar_states_MPPI[:,:,:,0],demo_x_min,demo_x_max))
                    radar_states_MPPI=radar_states_MPPI.at[:,:,:,1].set(jnp.clip(radar_states_MPPI[:,:,:,1],demo_y_min,demo_y_max))
                    

                    mppi_rollout_end = time()
                    

                    # GET MPC OBJECTIVE
                    mppi_score_start = time()
                    # Score all the rollouts
                    cost_trajectory = MPPI_scores(radar_state, target_states_rollout, V,
                                              A=A,J=J)
                    

                    mppi_score_end = time()
                    
                    #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                    cost_MPPI = args.alpha1*cost_trajectory

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
                radar_state=radar_state.at[:,0].set(jnp.clip(radar_state[:,0],demo_x_min,demo_x_max))
                radar_state=radar_state.at[:,1].set(jnp.clip(radar_state[:,1],demo_y_min,demo_y_max))
                
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

            J = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                      J=J,method="NN",state_train=state_train)

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

                thetas = jnp.arcsin(target_state_true[:, 2] / args.R2T)
                radius_projected = args.R2T * jnp.cos(thetas)

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
    
           
               
        return states,traj_probs,actions
    
    
    
    def generate_session_N_CIRL(self, NT,p0,chis, pt,chit,m0,A,time_steps,time_step_size,limits,MPPI_scores,MPPI_scores_t,state_train,epochs,IM_fn):
        gif_savepath = os.path.join( "images", "gifs")
        gif_savename =  f"JU_MPPI.gif"
        os.makedirs("tmp_images",exist_ok=True)
        images = []
        M,dm=m0.shape
        N,dn=p0.shape
        ps=p0
        gamma = self.gamma
        
        MPPI_iterations=self.MPPI_iterations
        states, traj_probs, actions,FIMs = [], [], [],[]
        stds = jnp.array([[-3,3],
                          [-45* jnp.pi/180, 45 * jnp.pi/180]])
        stds_t=stds.copy()
        num_traj=self.num_traj
        seed = self.seed
        key = jax.random.PRNGKey(seed)
        key_t=jax.random.PRNGKey(seed)
        u_ptb_method = "mixture"
        v_init = self.v_init
        av_init = self.av_init
        temperature = self.temperature
        
        U_V = jnp.ones((N,time_steps,1)) * v_init
        U_W = jnp.ones((N,time_steps,1)) * av_init
        U_Nom =jnp.concatenate((U_V,U_W),axis=-1)
        U_Nom_t=U_Nom.copy()
        J = jnp.eye(dm*M)
        
        state_multiple_update_vmap = vmap(state_multiple_update, (0, 0, 0, None))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        fig_debug,axes_debug = plt.subplots(1,1,figsize=(10,10))
        target_states=jnp.concatenate((pt,chit),axis=1)
        m0=m0.at[:,:2].set(pt)
        
        for k in range(NT):
            #print(f"\n Step {k} MPPI Iteration: ")
            qs_previous = m0
            #m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)
            key, subkey = jax.random.split(key)
        
            best_mppi_iter_score = -np.inf
            best_mppi_iter_score_t = -np.inf
            mppi_round_time_start = time()
            m0=m0.at[:,:2].set(pt)
        
            target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])
        
            for mppi_iter in range(MPPI_iterations):
                start = time()
                key, subkey = jax.random.split(key)
        
                mppi_start = time()
                U_ptb = MPPI_ptb(stds,N, time_steps, num_traj, key,method=u_ptb_method)
        
                mppi_rollout_start = time()
                U_MPPI,P_MPPI,CHI_MPPI, _,_,_ = MPPI(U_nominal=U_Nom, chis_nominal=chis,
                                                                   U_ptb=U_ptb,ps=ps,
                                                                   time_step_size=time_step_size, limits=limits)
                mppi_rollout_end = time()
    
                mppi_score_start = time()
                scores_MPPI = MPPI_scores(ps, target_states_rollout, U_MPPI, chis, time_step_size,
                                          A=A,J=J,
                                          gamma=gamma,state_train=state_train)
                mppi_score_end = time()
            
                scores_temp = -1/(temperature)*scores_MPPI
            
                max_idx = jnp.argmax(scores_temp)
                SCORE_BEST = scores_temp[max_idx]
        
                if SCORE_BEST > best_mppi_iter_score:
                    if k == 0:
                        print("First Iter Best Score: ",SCORE_BEST)
                    best_mppi_iter_score = SCORE_BEST
                    U_BEST = U_MPPI[max_idx]
        
                    # print(SCORE_BEST)
        
                scores_MPPI_weight = jax.nn.softmax(scores_temp)
        
        
                delta_actions = U_MPPI - U_Nom
                # U_Nom = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)
                U_Nom += jnp.sum(delta_actions * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)
        
                mppi_end = time()
        
            mppi_round_time_end = time()
            U_Nom = jnp.roll(U_BEST,-1,axis=1)
            if (MPPI_iterations ==50):
                print("MPPI Round Time: ",mppi_round_time_end-mppi_round_time_start)
                print("MPPI Iter Time: ",mppi_end-mppi_start)
                print("MPPI Score Time: ",mppi_score_end-mppi_score_start)
                print("MPPI Mean Score: ",-jnp.nanmean(scores_MPPI))
                print("MPPI Best Score: ",best_mppi_iter_score)
                # FIMs.append(-jnp.nanmean(scores_MPPI))
        
        
        
            # U_BEST =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            # U_nominal =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_vmap(jnp.expand_dims(ps, 1), U_BEST ,
                                                                           chis, time_step_size)
            
            
            ##target MPPI 
            #key_t, subkey_t = jax.random.split(key_t)
            for mppi_iter in range(MPPI_iterations):
                start = time()
                key_t, subkey_t = jax.random.split(key_t)
        
                mppi_start = time()
                U_ptb_t = MPPI_ptb(stds_t,N, time_steps, num_traj, key_t,method=u_ptb_method)
        
                mppi_rollout_start = time()
                U_MPPI_T,P_MPPI_T,CHI_MPPI_T, _,_,_ = MPPI(U_nominal=U_Nom_t, chis_nominal=chit,
                                                                    U_ptb=U_ptb_t,ps=pt,
                                                                    time_step_size=time_step_size, limits=limits)
                mppi_rollout_end = time()
    
                mppi_score_start = time()
                scores_MPPI_T = MPPI_scores_t(Sensor_Positions, target_states, U_MPPI_T, Sensor_Chis, time_step_size,
                                          A=A,J=J,
                                          gamma=gamma,state_train=state_train)
                mppi_score_end = time()
            
                scores_temp_T = -1/(temperature)*scores_MPPI_T
            
                max_idx_t = jnp.argmax(scores_temp_T)
                SCORE_BEST_T = scores_temp_T[max_idx_t]
        
                if SCORE_BEST_T > best_mppi_iter_score_t:
                    if k == 0:
                        print("First Iter Best Score: ",SCORE_BEST_T)
                    best_mppi_iter_score_t = SCORE_BEST_T
                    U_BEST_T = U_MPPI_T[max_idx_t]
                    #U_BEST_T = jnp.clip(U_BEST_T,-50,50)
        
                    # print(SCORE_BEST)
        
                scores_MPPI_weight_T = jax.nn.softmax(scores_temp_T)
        
        
                delta_actions_t = U_MPPI_T - U_Nom_t
                # U_Nom = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)
                U_Nom_t += jnp.sum(delta_actions_t * scores_MPPI_weight_T.reshape(-1, 1, 1, 1), axis=0)
        
                mppi_end = time()
        
            mppi_round_time_end = time()
            U_Nom_t = jnp.roll(U_BEST_T,-1,axis=1)

        
        
        
            # U_BEST =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            #U_nominal =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            _, _, Target_Positions, Target_Chis = state_multiple_update_vmap(jnp.expand_dims(pt, 1), U_BEST_T ,
                                                                            chit, time_step_size)
            pt=Target_Positions[:,1]
            chit=Target_Chis[:,1]
            target_states=jnp.concatenate((pt,chit),axis=1)
            sign_vel=np.sign(pt-qs_previous[:,:2])
            A=A.at[3,3].set(1* sign_vel[0][0])
            A=A.at[4,4].set(1* sign_vel[0][1])
            
            
            cov=jnp.array([[stds[0,1]**2,0],
                              [0, stds[0,1]**2]])
            probs_fun=vmap(self.predict_probs,(0,None,0))
            prob=self.predict_probs(U_Nom[:,0,:], cov, U_BEST[:,0,:])*self.predict_probs(U_Nom_t[:,0,:], cov, U_BEST_T[:,0,:])
            #prob=np.prod(probs_fun(U_Nom[:,0,:], cov, U_BEST[:,0,:]))
            #prob=probs_fun(U_Nom[:,0,:], cov, U_BEST[:,0,:])
           
            # if k == 0:
            #     MPPI_visualize(P_MPPI, Sensor_Positions)
            # print(ps.shape,chis.shape,ps.squeeze().shape)
            ps = Sensor_Positions[:,1,:]
            chis = Sensor_Chis[:,1]
            #Sensor_Positions = np.asarray(ps)
            pos=jnp.concatenate((ps,jnp.zeros((ps.shape[0],1))),-1)
            state=jnp.concatenate((pos.flatten(),m0[:,0:3].flatten()))
            
            states.append(state)
            traj_probs.append(prob.flatten())
            actions.append(U_BEST[:,0,:].flatten())
            J = IM_fn(radar_states=ps,target_states=m0,J=J,actions=None,state_train=None) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]
             # print(jnp.trace(J))
            FIMs.append(jnp.linalg.slogdet(J)[1].ravel())
            
            if (epochs %3 ==0):
                file = os.path.join("tmp_images",f"JU_test_target_movement{k}.png")
                fig_time = time()
                # for n in range(N):
                #     axes[0].plot(P_MPPI[:,n,:,0].T,P_MPPI[:,n,:,1].T,'b-',label="__nolegend__")
                axes[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
                axes[0].plot(m0[:,0], m0[:,1], 'go',label="Target Position")
                axes[0].plot(p0[:,0], p0[:,1], 'md',label="Sensor Init")
               
    
    
                
                axes[0].plot(Sensor_Positions[:,0,0], Sensor_Positions[:,0,1], 'r*',label="Sensor Position")
                axes[0].plot(Sensor_Positions[:,1:,0].T, Sensor_Positions[:,1:,1].T, 'r-',label="_nolegend_")
                axes[0].plot([],[],"r.-",label="Sensor Planned Path")
                axes[0].set_title(f"k={k}")
                # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
                axes[0].legend(bbox_to_anchor=(0.7, 1.45),loc="upper center")
    
                
                

                axes[1].plot(jnp.array(FIMs),'ko')
                axes[1].set_ylabel("LogDet FIM (Higher is Better)")
                axes[1].set_title(f"Avg MPPI LogDet FIM={np.round(FIMs[-1])}")
                fig.tight_layout()
                fig.savefig(file)

                axes[0].cla()
                axes[1].cla()
                fig_time = time()-fig_time
                images.append(file)

        if (epochs %3 ==0):
            images = [imageio.imread(file) for file in images]
            imageio.mimsave(os.path.join(gif_savepath,gif_savename),images,duration=0.1)#  
            
        return states,traj_probs,actions
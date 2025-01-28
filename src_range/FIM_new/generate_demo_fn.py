from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp

from sklearn.covariance import OAS

from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,JU_RANGE_SFIM,Single_FIM_Radar,FIM_Visualization
from src_range.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range.utils import visualize_tracking,visualize_control,visualize_target_mse,place_sensors_restricted
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

from cost_jax import CostNN, apply_model, update_model
from flax.training import train_state,checkpoints
import flax 
import optax
import scipy 
from tqdm import tqdm

config.update("jax_enable_x64", True)


def generate_demo_MPPI(state_train):

    
    seed = 555
    key = jax.random.PRNGKey(seed)

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"

    # Save frames as a GIF
    gif_filename = "radar_optimal_RICE.gif"
    gif_savepath = os.path.join("..", "..", "images","gifs")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = .1
    NT = 300
    N = 1

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    K = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    coef = K / (R ** 4)
    Pr = K / (R ** 4)

    # get the power of the noise of the signalf
    SNR=0

    SCNR = -20
    CNR = -10
    Pt = 10000
    Amp, Ma, zeta, s = NoiseParams(Pt * coef, SCNR, CNR=CNR)

    print("Spread: ",s**2)
    print("Power Return (RCS): ",coef*Pt)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(Pt))
    print("lam ={:.9f}".format(lam))

    key_args = {"Pt": Pt, "Gt": Gt, "Gr": Gr, "lam": lam, "L": L, "rcs": rcs, "R": 100,"SCNR":SCNR,"CNR":CNR,"s":s}

    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 10
    R_sensors_to_targets = 5.
    R_sensors_to_sensors = 1.5
    time_step_size = T
    max_velocity = 50.
    min_velocity = 0
    max_angle_velocity = jnp.pi
    min_angle_velocity = -jnp.pi

    # ==================== MPPI CONFIGURATION ================================= #
    limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])

    # ps = place_sensors([-100,100],[-100,100],N)
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)
    ps_init = deepcopy(ps)
    z_elevation=10
    qs = jnp.array([[0.0, -0.0, z_elevation,25., 20,0]])
    # qs = jnp.array([[0.0, -0.0, 25., 20], #,#,
    #                 [-50.4,30.32,-20,-10], #,
    #                 [10,10,10,10],
    #                 [20,20,5,-5]])
    pt=qs[:,:2]
    chit = jax.random.uniform(key,shape=(pt.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))

    M, d = qs.shape;
    N = len(ps);
    dn=2

    # ======================== MPC Assumptions ====================================== #
    gamma = 0.3
    paretos = jnp.ones((M,)) * 1 / M  # jnp.array([1/3,1/3,1/3])
    #paretos = jnp.array([1,0,0,0])

    assert len(paretos) == M, "Pareto weights not equal to number of targets!"
    assert (jnp.sum(paretos) <= (1 + 1e-5)) and (jnp.sum(paretos) >= -1e-5), "Pareto weights don't sum to 1!"


    sigmaQ = np.sqrt(10 ** 2);
    sigmaV = jnp.sqrt(9)

    print("SigmaQ (state noise)={}".format(sigmaQ))

    # A_single = jnp.array([[1., 0, T, 0],
    #                       [0, 1., 0, T],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1.]])
    #
    # Q_single = jnp.array([
    #     [(T ** 4) / 4, 0, (T ** 3) / 2, 0],
    #     [0, (T ** 4) / 4, 0, (T ** 3) / 2],
    #     [(T ** 3) / 2, 0, (T ** 2), 0],
    #     [0, (T ** 3) / 2, 0, (T ** 2)]
    # ]) * sigmaQ ** 2

    A_single = jnp.array([[1., 0, 0, T, 0, 0],
                   [0, 1., 0, 0, T, 0],
                   [0, 0, 1, 0, 0, T],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
        [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
        [0, 0, (T**4)/4, 0, 0, (T**3) / 2],
        [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
        [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
        [0, 0, (T**3) / 2, 0, 0, (T**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = jnp.eye(N)

    nx = Q.shape[0]

    Js = [jnp.eye(d) for m in range(M)]

    JU_FIM_D_Radar(ps, q=qs[[0],:], J=Js[0],U=0,
                   A=A_single, Q=Q_single,
                   Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(JU_FIM_radareqn_target_logdet)

    print("Optimization START: ")
    lbfgsb =  ScipyBoundedMinimize(fun=Multi_FIM_Logdet, method="L-BFGS-B",jit=True)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    time_step_sizes = jnp.tile(time_step_size, (N, 1))

    U_upper = (jnp.ones((time_steps, 2)) * jnp.array([[max_velocity, max_angle_velocity]]))
    U_lower = (jnp.ones((time_steps, 2)) * jnp.array([[min_velocity, min_angle_velocity]]))

    U_lower = jnp.tile(U_lower, jnp.array([N, 1, 1]))
    U_upper = jnp.tile(U_upper, jnp.array([N, 1, 1]))
    bounds = (U_lower, U_upper)

    m0 = qs

    J_list = []
    traj_list=[]
    traj_sindy_list=[]
    frames = []
    frame_names = []
    state_multiple_update_parallel = vmap(state_multiple_update, (0, 0, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_velocity+5, maxval=max_velocity)
    U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_angle_velocity,maxval=max_angle_velocity)
    U = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)
    
    method = "Single_FIM_3D_action_MPPI_NN"
    method_t = "Single_FIM_3D_action_MPPI_NN_t"
    #MPPI_method = "single"
    MPPI_method = "NN"
    MPPI_method_t="NN_t"
    sigmaW = jnp.sqrt(M*Pr/ (10**(SNR/10)))
    
    MPPI_iterations=10
    states, traj_probs, actions,FIMs = [], [], [],[]
    stds = jnp.array([[-3,3],
                      [-45* jnp.pi/180, 45 * jnp.pi/180]])
    stds_t=stds.copy()
    num_traj=500
    key = jax.random.PRNGKey(seed)
    key_t=jax.random.PRNGKey(seed)
    u_ptb_method = "mixture"
    temperature = 0.1
    v_init = 0
    av_init = 0
    U_V = jnp.ones((N,time_steps,1)) * v_init
    U_W = jnp.ones((N,time_steps,1)) * av_init
    U_Nom =jnp.concatenate((U_V,U_W),axis=-1)
    U_Nom_t =jnp.concatenate((U_V,U_W),axis=-1)


    # ==================== MPPI CONFIGURATION ================================= #
    limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])
    dm=d
    Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

    IM_fn = partial(Single_JU_FIM_Radar_MPPI,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW,method=MPPI_method)
    IM_fn_t = partial(Single_JU_FIM_Radar_MPPI,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW,method=MPPI_method_t)
    #IM_fn(ps,qs,J=J,actions=U_Nom)
    IM_fn_GT = partial(Single_JU_FIM_Radar_MPPI,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW)

    Multi_FIM_Logdet_v = Multi_FIM_Logdet_decorator_MPPI(IM_fn=IM_fn,method=method)
    Multi_FIM_Logdet_t = Multi_FIM_Logdet_decorator_MPPI(IM_fn=IM_fn_t,method=method_t)
    MPPI_scores = MPPI_scores_wrapper(Multi_FIM_Logdet_v,method=MPPI_method)
    MPPI_scores_t = MPPI_scores_wrapper(Multi_FIM_Logdet_t,method=MPPI_method)
    
    J = jnp.eye(dm*M)
    target_states=jnp.concatenate((pt,chit),axis=1)

    m0=m0.at[:,:2].set(pt)
    state_multiple_update_vmap = vmap(state_multiple_update, (0, 0, 0, None))
    
    for k in range(NT):
        start = time()
        qs_previous = m0

        #m0 = (A @ m0.reshape(-1, 1)).reshape(M, d)

        

        # U = jnp.zeros((N,2,time_steps))

        U = lbfgsb.run(U, bounds=bounds, chis=chis, ps=ps, qs=m0,
                       time_step_sizes=time_step_sizes,
                       Js=Js, paretos=paretos,
                       A=A_single, Q=Q_single,
                       Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,B=B,c=c,alpha=alpha,
                       gamma=gamma,
                       ).params

        _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_parallel(ps,
                                                                                            U,
                                                                                            chis, time_step_sizes)
        ps = Sensor_Positions[:,1,:]
        chis = Sensor_Chis[:,1,:]
        
        # MPPI 
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
        _, _, Sensor_Positions_v, Sensor_Chis_v = state_multiple_update_vmap(jnp.expand_dims(ps, 1), U_BEST ,
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
            scores_MPPI_T = MPPI_scores_t(Sensor_Positions_v, target_states, U_MPPI_T, Sensor_Chis_v, time_step_size,
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
        A_single=A_single.at[3,3].set(1* sign_vel[0][0])
        A_single=A_single.at[4,4].set(1* sign_vel[0][1])
        
        qs=qs.at[:,:2].set(pt)
        
        

        # print(ps.shape,chis.shape,ps.squeeze().shape)
        # ps = ps.squeeze()
        # chis = chis.squeeze()


        end = time()
        print(f"Step {k} Optimization Time: ",end-start)

        # m0  = ps

        Js = [JU_FIM_D_Radar(ps=ps, q=m0[[i],:],U=0, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],B=B,c=c,alpha=alpha) for i in range(len(Js))]
       # Jt=JU_FIM_D_Radar(ps=ps[0,:], q=m0, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[0],B=B,c=c,alpha=alpha)
        traj=np.concatenate((ps.flatten(),qs[:,:3].flatten(),U[:,0,:].flatten()),axis=0)
        traj_list.append(traj)
        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        #J_list.append([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        J_list.append([jnp.log(Ji)[1].item() for Ji in Js])
        print("FIM (higher is better) ",np.sum(J_list[-1]))

        save_time = time()
        if (k+1)%frame_skip == 0:
            plt.minorticks_off()

            axes[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="_nolegend_")
            axes[0].plot(m0[:,0], m0[:,1], 'go',label="Targets")
            axes[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")
            axes[0].plot(ps[:,0], ps[:,1], 'rx',label="Sensors")
            axes[0].plot(Sensor_Positions[:,1:,0].T, Sensor_Positions[:,1:,1].T, 'r.-',label="_nolegend_")
            axes[0].plot([],[],"r.-",label="Sensor Planned Path")

            axes[0].legend(bbox_to_anchor=(0.7, 0.95))
            axes[0].set_title(f"k={k}")

            qx,qy,logdet_grid = FIM_Visualization(ps=ps, qs=m0,
                                                Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW,
                                                N=1000)

            axes[1].contourf(qx, qy, logdet_grid, levels=20)
            axes[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes[1].set_title("Instant Time Objective Function Map")

            axes[2].plot(jnp.sum(jnp.array(J_list),axis=1))
            axes[2].set_ylabel("sum of individual target logdet FIM")
            axes[2].set_xlabel("Time Step")


            filename = f"frame_NCIRL_{k}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(photo_dump, filename))

            frame_names.append(os.path.join(photo_dump, filename))
            axes[0].cla()
            axes[1].cla()
            axes[2].cla()
        save_time = time() - save_time
        print("Figure Save Time: ",save_time)
    
    arr=np.zeros(300)
    for i in range (len(J_list)):
    
        a=J_list[i][0]
        arr[i]=a
    plt.figure()
    plt.plot(jnp.array(J_list))
    plt.show()
    print("lol")
    
    
    
    return traj_list 


def generate_demo_MPPI_single(args,state_train):

    
       mpl.rcParams['path.simplify_threshold'] = 1.0
       mplstyle.use('fast')
       mplstyle.use(['ggplot', 'fast'])
       key = jax.random.PRNGKey(args.seed)
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
       # cov = jnp.stack([cov_traj for n in range(N)])

       mpc_method = "Single_FIM_3D_action_MPPI"

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

      

       if args.fim_method_demo == "PFIM":
           IM_fn = partial(Single_JU_FIM_Radar, A=A, Q=Q, C=C)
           IM_fn_update = partial(Single_JU_FIM_Radar, A=A_ckf, Q=Q_ckf, C=C)
       elif args.fim_method_demo == "SFIM":
           IM_fn = partial(Single_FIM_Radar, C=C)
           IM_fn_update = IM_fn
       elif args.fim_method_demo == "SFIM_bad":
           sigmaR = 1
           IM_fn = partial(JU_RANGE_SFIM, R=jnp.eye(M_target*args.N_radar)*sigmaR**2)
           IM_fn_update = IM_fn


       MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method)

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
       pbar = tqdm(total=args.N_steps, desc="Starting")

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

               #print(f"\n Step {step} MPPI CONTROL ")


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
                    #   if mppi_iter == 0:
                          # print("Oracle Approx Shrinkage: ",np.round(oas.shrinkage_,5))

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
               radar_states = kinematic_model(U,
                                              radar_state, args.dt_control)

               #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)

               radar_state = radar_states[:,1]
               U = jnp.roll(U, -1, axis=1)
               ps=radar_state[:,:3]

               mppi_end_time = time()
               #print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))





           J = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                     J=J)
           traj=np.concatenate((ps.flatten(),target_state_true[:,:3].flatten(),U[:,0,:].flatten()),axis=0)
           traj_sindy=np.concatenate((ps,target_state_true[:,:3]),axis=0)
           traj_list.append(traj)
           traj_sindy_list.append(traj_sindy)
           # print(jnp.linalg.slogdet(J)[1].ravel().item())
           if args.N_radar==1 and M_target==1:
               FIMs[step] = jnp.log(J)
           else:
               
               FIMs[step] = jnp.linalg.slogdet(J)[1].ravel().item()

           radar_state_history[step] = radar_state
           #print("FIM :" , FIMs[step])
           pbar.set_description(
               f"FIM_demo = {FIMs[step]} ")
           pbar.update(1)

           if args.save_images and (step % 4 == 0):
               #print(f"Step {step} - Saving Figure ")

               axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                        label="Radar Initial Position")

               thetas = jnp.arcsin(target_state_true[:, 2] / args.R2T)
               radius_projected = args.R2T * jnp.cos(thetas)

              # print("Target Height :",target_state_true[:,2])
               #print("Radius Projected: ",radius_projected)

               # radar_state_history[step] = radar_state

               try:
                   imgs_main.append(visualize_tracking(target_state_true=target_state_true, target_state_ckf=target_state_true,target_states_true=target_states_true.T.reshape(-1,M_target,dm)[:step],
                              radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,radar_state_history=radar_state_history[max(step // update_freq_control - args.tail_length,0):step // update_freq_control],
                              cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                              R2T=args.R2T, R2R=args.R2R,C=C,
                              fig=fig_main, axes=axes_main, step=step,
                              tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI"))
               except Exception as error:
                   print("Tracking Img Could Not save: ",error)

              
               try:
                   imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                              fig=fig_control, axes=axes_control, step=step,
                              tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_control"))
               except:
                   print("Control Img Could Not Save")


           # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)

           # CKF ! ! ! !
          

       pbar.close()
       np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')

       # if args.save_images:
       #     visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

       #     images = [imageio.imread(file) for file in imgs_main]
       #     imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method_demo}.gif'), images, duration=0.1)

       #     images = [imageio.imread(file) for file in imgs_control]
       #     imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


       #     if args.remove_tmp_images:
       #         shutil.rmtree(args.tmp_img_savepath)
    
       return traj_list , traj_sindy_list,FIMs
   

def generate_demo_MPPI_NCIRL(args,state_train):

    
       mpl.rcParams['path.simplify_threshold'] = 1.0
       mplstyle.use('fast')
       mplstyle.use(['ggplot', 'fast'])
       key = jax.random.PRNGKey(args.seed)
       key_t = jax.random.PRNGKey(args.seed)
       key_v = jax.random.PRNGKey(args.seed)
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
       cov_v = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(args.N_radar)])
       # cov = jnp.stack([cov_traj for n in range(N)])

       mpc_method = "Single_FIM_3D_action_MPPI"
       mpc_method_v = "Single_FIM_3D_action_NN_MPPI"
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
       pt=target_state_v[:,:3]
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
        
       

       MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method)

       MPPI_scores = MPPI_scores_wrapper(MPC_obj)

       MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)
       
       MPC_obj_v = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method_v,state_train=state_train)

       MPPI_scores_v = MPPI_scores_wrapper(MPC_obj_v)

       MPPI_v= MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)
       
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
       U_v =jnp.concatenate((U1,U2),axis=-1)

       if not args.move_radars:
           U = jnp.zeros_like(U)
           U_t = jnp.zeros_like(U_t)
           U_v = jnp.zeros_like(U_v)
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


        


               ## virtual cost function movemement
               
               U_prime_v = deepcopy(U_v)
               cov_prime_v = deepcopy(cov_v)
               for mppi_iter in range(args.MPPI_iterations):
                  start = time()
                  key_v, subkey_v = jax.random.split(key_v)

                  mppi_start = time()

                  E_v = jax.random.multivariate_normal(key_v, mean=jnp.zeros_like(U_v).ravel(), cov=cov_prime_v, shape=(args.num_traj,),method="svd")

                  # simulate the model with the trajectory noise samples
                  V_v = U_prime_v + E_v.reshape(args.num_traj,args.N_radar,args.horizon,2)

                  mppi_rollout_start = time()

                  radar_states_v,radar_states_MPPI_v = MPPI_v(U_nominal=U_prime_v,
                                                                     U_MPPI=V_v,radar_state=radar_state)

                  mppi_rollout_end = time()
                  

                  # GET MPC OBJECTIVE
                  mppi_score_start = time()
                  # Score all the rollouts
                  cost_trajectory_v = MPPI_scores_v(radar_state, target_states_rollout, V_v,
                                            A=A,J=J)
                  
                  cost_collision_r2t_v = collision_penalty_vmap(radar_states_MPPI_v[...,1:args.horizon+1,:], target_states_rollout,
                                          args.x_min,args.x_max,args.y_min,args.y_max)

                  cost_collision_r2t_v = jnp.sum((cost_collision_r2t_v * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                   
      
                  mppi_score_end = time()
                  
                  #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                  cost_MPPI_v = args.alpha1*cost_trajectory_v+ args.alpha2*cost_collision_r2t_v

                  weights_v = weight_fn(cost_MPPI_v)


                  if jnp.isnan(cost_MPPI_v).any():
                      print("BREAK!")
                      break

                  if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):

                      best_cost_v = jnp.sum(cost_MPPI_v*weights_v)

                      U_copy_v = deepcopy(U_prime_v)
                      U_prime_v = U_prime_v + jnp.sum(weights_v.reshape(args.num_traj,1,1,1) * E_v.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

                      oas_v = OAS(assume_centered=True).fit(E_v[weights_v != 0])
                      cov_prime_v = jnp.array(oas_v.covariance_)
                      if mppi_iter == 0:
                          print("Oracle Approx Shrinkage virtual: ",np.round(oas_v.shrinkage_,5))

               mppi_round_time_end = time()

               if jnp.isnan(cost_MPPI_v).any():
                  print("BREAK!")
                  break

               weights_v = weight_fn(cost_MPPI_v)

               mean_shift_v = (U_prime_v - U_v)

               E_prime_v = E_v + mean_shift_v.ravel()

               U_v += jnp.sum(weights_v.reshape(-1,1,1,1) * E_prime_v.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

               U_v = jnp.stack((jnp.clip(U_v[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U_v[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)

              # jnp.repeat(U,update_freq_control,axis=1)

              # radar_states = kinematic_model(U ,radar_state, dt_control)

              # generate radar states at measurement frequency
               radar_states = kinematic_model(U,
                                             radar_state, args.dt_control)

              #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)

               radar_state = radar_states[:,1]
               U = jnp.roll(U, -1, axis=1)
               ps=radar_state[:,:3]

              
                
              # generate radar states at measurement frequency
               radar_states_v = kinematic_model(U_v,
                                             radar_state, args.dt_control)

              #U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)

               radar_state_v = radar_states_v[:,1]
               U_v = jnp.roll(U_v, -1, axis=1)
               ps_v=radar_state_v[:,:3]

               mppi_end_time = time()
               print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))
              
                
              
              ## target cost function movement
               U_prime_t = deepcopy(U_t)
               cov_prime_t = deepcopy(cov_t)
       
               print(f"\n Step {step} MPPI CONTROL TARGET ")
               
               
               for mppi_iter in range(args.MPPI_iterations):
                   start = time()
                   key_t, subkey = jax.random.split(key_t)
       
                   mppi_start = time()
       
                   E_t = jax.random.multivariate_normal(key_t, mean=jnp.zeros_like(U_t).ravel(), cov=cov_prime_t, shape=(args.num_traj,),method="svd")
       
                   # simulate the model with the trajectory noise samples
                   V_t = U_prime_t + E_t.reshape(args.num_traj,args.N_radar,args.horizon,2)
       
                   mppi_rollout_start = time()
       
                   target_states,target_states_MPPI = MPPI(U_nominal=U_prime_t,
                                                                      U_MPPI=V_t,radar_state=target_state)
       
                   mppi_rollout_end = time()
       
       
                   # GET MPC OBJECTIVE
                   mppi_score_start = time()
                   # Score all the rollouts
                   cost_trajectory_t = MPPI_scores_t(radar_states_v, target_state, V_t,
                                             A=A,J=J)
       
                   mppi_score_end = time()
                   cost_collision_r2t_t = collision_penalty_vmap(target_states_MPPI[...,1:args.horizon+1,:], radar_states_v,
                                           args.x_min,args.x_max,args.y_min,args.y_max)

                   cost_collision_r2t_t = jnp.sum((cost_collision_r2t_t * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                    
       
                   
                   #cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed
                   cost_MPPI_t = args.alpha1*cost_trajectory_t + args.alpha2*cost_collision_r2t_t
       
                   weights_t = weight_fn(cost_MPPI_t)
       
       
                   if jnp.isnan(cost_MPPI_t).any():
                       print("BREAK!")
                       break
       
                   if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):
       
                       best_cost_t = jnp.sum(cost_MPPI_t*weights_t)
       
                       U_copy_t = deepcopy(U_prime_t)
                       U_prime_t = U_prime_t + jnp.sum(weights_t.reshape(args.num_traj,1,1,1) * E_t.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
       
                       oas_t = OAS(assume_centered=True).fit(E_t[weights_t != 0])
                       cov_prime_t = jnp.array(oas_t.covariance_)
                       if mppi_iter == 0:
                           print("Oracle Approx Shrinkage: ",np.round(oas_t.shrinkage_,5))
               
               mppi_round_time_end = time()
       
               if jnp.isnan(cost_MPPI_t).any():
                   print("BREAK!")
                   break
       
               weights_t = weight_fn(cost_MPPI_t)
       
               mean_shift_t = (U_prime_t - U_t)
       
               E_prime_t = E_t + mean_shift_t.ravel()
       
               U_t += jnp.sum(weights_t.reshape(-1,1,1,1) * E_prime_t.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)
       
               U_t = jnp.stack((jnp.clip(U_t[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U_t[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)
       
               # jnp.repeat(U,update_freq_control,axis=1)
       
               target_states = kinematic_model(U_t ,target_state, args.dt_control)
       
               # generate radar states at measurement frequency
               #target_states = kinematic_model(np.repeat(U_t, update_freq_control, axis=1)[:, :update_freq_control, :],
               #                              target_state, args.dt_control)
       
               # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)
       
               target_state = target_states[:,1]
               pt=target_state[:,:3]
               U_t = jnp.roll(U_t, -1, axis=1)
       
               mppi_end_time = time()
               print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))     
               
               
              
           target_state_v=jnp.concatenate((pt,target_state_v[:,3:]),axis=1)
           J = IM_fn_update(radar_state=radar_state, target_state=target_state_v,
                     J=J)
           traj=np.concatenate((ps.flatten(),target_state[:,:3].flatten(),U[:,0,:].flatten()),axis=0)
           traj_sindy=np.concatenate((ps,target_state[:,:3]),axis=0)
           traj_list.append(traj)
           traj_sindy_list.append(traj_sindy)
           # print(jnp.linalg.slogdet(J)[1].ravel().item())
           if args.N_radar==1 and M_target==1:
               FIMs[step] = jnp.log(J)
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
                   imgs_main.append(visualize_tracking(target_state_true=target_state_v, target_state_ckf=target_state_v,target_states_true=None,
                              radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,radar_state_history=radar_state_history[max(step // update_freq_control - args.tail_length,0):step // update_freq_control],
                              cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                              R2T=args.R2T, R2R=args.R2R,C=C,
                              fig=fig_main, axes=axes_main, step=step,
                              tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI"))
               except Exception as error:
                   print("Tracking Img Could Not save: ",error)

              
               try:
                   imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                              fig=fig_control, axes=axes_control, step=step,
                              tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_control"))
               except:
                   print("Control Img Could Not Save")


           # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)

           # CKF ! ! ! !
          


       np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')

       if args.save_images:
           visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

           images = [imageio.imread(file) for file in imgs_main]
           imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method_demo}.gif'), images, duration=0.1)

           images = [imageio.imread(file) for file in imgs_control]
           imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


           if args.remove_tmp_images:
               shutil.rmtree(args.tmp_img_savepath)
    
       return traj_list , traj_sindy_list


def generate_demo_MPC(state_train):
    
    seed = 555
    key = jax.random.PRNGKey(seed)

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"

    # Save frames as a GIF
    gif_filename = "radar_optimal_RICE.gif"
    gif_savepath = os.path.join("..", "..", "images","gifs")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = .1
    NT = 300
    N = 1

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    K = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    coef = K / (R ** 4)
    Pr = K / (R ** 4)

    # get the power of the noise of the signalf
    SNR=0

    SCNR = -20
    CNR = -10
    Pt = 10000
    Amp, Ma, zeta, s = NoiseParams(Pt * coef, SCNR, CNR=CNR)

    print("Spread: ",s**2)
    print("Power Return (RCS): ",coef*Pt)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(Pt))
    print("lam ={:.9f}".format(lam))

    key_args = {"Pt": Pt, "Gt": Gt, "Gr": Gr, "lam": lam, "L": L, "rcs": rcs, "R": 100,"SCNR":SCNR,"CNR":CNR,"s":s}

    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 10
    R_sensors_to_targets = 5.
    R_sensors_to_sensors = 1.5
    time_step_size = T
    max_velocity = 50.
    min_velocity = 0
    max_angle_velocity = jnp.pi
    min_angle_velocity = -jnp.pi

    # ==================== MPPI CONFIGURATION ================================= #
    limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])

    # ps = place_sensors([-100,100],[-100,100],N)
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)
    ps_init = deepcopy(ps)
    z_elevation=10
    qs = jnp.array([[0.0, -0.0, z_elevation,25., 20,0]])
    # qs = jnp.array([[0.0, -0.0, 25., 20], #,#,
    #                 [-50.4,30.32,-20,-10], #,
    #                 [10,10,10,10],
    #                 [20,20,5,-5]])
    qs_init = deepcopy(qs)
    pt=qs[:,:2]
    chit = jax.random.uniform(key,shape=(pt.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    m0=qs
    
    M, d = qs.shape;
    N = len(ps);
    dn=2

    # ======================== MPC Assumptions ====================================== #
    gamma = 0.3
    paretos = jnp.ones((M,)) * 1 / M  # jnp.array([1/3,1/3,1/3])
    #paretos = jnp.array([1,0,0,0])

    assert len(paretos) == M, "Pareto weights not equal to number of targets!"
    assert (jnp.sum(paretos) <= (1 + 1e-5)) and (jnp.sum(paretos) >= -1e-5), "Pareto weights don't sum to 1!"


    sigmaQ = np.sqrt(10 ** 2);
    sigmaV = jnp.sqrt(9)

    print("SigmaQ (state noise)={}".format(sigmaQ))

    # A_single = jnp.array([[1., 0, T, 0],
    #                       [0, 1., 0, T],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1.]])
    #
    # Q_single = jnp.array([
    #     [(T ** 4) / 4, 0, (T ** 3) / 2, 0],
    #     [0, (T ** 4) / 4, 0, (T ** 3) / 2],
    #     [(T ** 3) / 2, 0, (T ** 2), 0],
    #     [0, (T ** 3) / 2, 0, (T ** 2)]
    # ]) * sigmaQ ** 2

    A_single = jnp.array([[1., 0, 0, T, 0, 0],
                   [0, 1., 0, 0, T, 0],
                   [0, 0, 1, 0, 0, T],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
        [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
        [0, 0, (T**4)/4, 0, 0, (T**3) / 2],
        [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
        [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
        [0, 0, (T**3) / 2, 0, 0, (T**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = jnp.eye(N)

    nx = Q.shape[0]

    Js = [jnp.eye(d) for m in range(M)]

    JU_FIM_D_Radar(ps, q=qs[[0],:], J=Js[0],U=0,
                   A=A_single, Q=Q_single,
                   Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(JU_FIM_radareqn_target_logdet)
    Multi_FIM_Logdet_v = Multi_FIM_Logdet_decorator_MPC(JU_FIM_radareqn_target_logdet,method="action_NN")
    Multi_FIM_target_Logdet = Multi_FIM_Logdet_decorator_MPC(JU_FIM_radareqn_target_logdet,method="Evasion_uni_NN")

    print("Optimization START: ")
    lbfgsb =  ScipyBoundedMinimize(fun=Multi_FIM_Logdet, method="L-BFGS-B",jit=True)
    lbfgsb_v =  ScipyBoundedMinimize(fun=Multi_FIM_Logdet_v, method="L-BFGS-B",jit=True)
    lbfgsb_target =  ScipyBoundedMinimize(fun=Multi_FIM_target_Logdet, method="L-BFGS-B",jit=True)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    time_step_sizes = jnp.tile(time_step_size, (N, 1))

    U_upper = (jnp.ones((time_steps, 2)) * jnp.array([[max_velocity, max_angle_velocity]]))
    U_lower = (jnp.ones((time_steps, 2)) * jnp.array([[min_velocity, min_angle_velocity]]))

    U_lower = jnp.tile(U_lower, jnp.array([N, 1, 1]))
    U_upper = jnp.tile(U_upper, jnp.array([N, 1, 1]))
    bounds = (U_lower, U_upper)

    

    J_list = []
    traj_list=[]
    frames = []
    frame_names = []
    state_multiple_update_parallel = vmap(state_multiple_update, (0, 0, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dm=6
    
    J = jnp.eye(dm*M)
    target_states=jnp.concatenate((pt,chit),axis=1)

    m0=m0.at[:,:2].set(pt)
    state_multiple_update_vmap = vmap(state_multiple_update, (0, 0, 0, None))
    state_target_multiple_update_parallel = vmap(state_target_multiple_update_uni, (0, 0, 0, 0))
    
    for k in range(NT):
        start = time()
        m0=m0.at[:,:2].set(pt)
        qs_previous = m0

        #m0 = (A @ m0.reshape(-1, 1)).reshape(M, d)
        U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_velocity, maxval=max_velocity)
        U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_angle_velocity,maxval=max_angle_velocity)
        U = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)
        

        # U = jnp.zeros((N,2,time_steps))
        #sensor original MPC
        U = lbfgsb.run(U, bounds=bounds, chis=chis, ps=ps, qs=m0,
                       time_step_sizes=time_step_sizes,
                       Js=Js, paretos=paretos,
                       A=A_single, Q=Q_single,
                       Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,B=B,c=c,alpha=alpha,
                       gamma=gamma,
                       ).params

        _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_parallel(ps,
                                                                                            U,
                                                                                            chis, time_step_sizes)
        
        
        #sensor virtual MPC
        U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_velocity, maxval=max_velocity)
        U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_angle_velocity,maxval=max_angle_velocity)
        U_acceleration = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)
       # U_acceleration = jax.random.uniform(key, shape=(M, time_steps, 3 ), minval=min_acceleration, maxval=max_acceleration)
        U_v = U_acceleration
        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])
        pt_trajectory=target_states_rollout[:,:,:2].transpose(1,0,2)
    
        U_v = lbfgsb_v.run(U_v, bounds=bounds, chis=chis, ps=ps, pt_trajectory=pt_trajectory,
                       time_step_sizes=time_step_sizes,
                       Js=Js, paretos=paretos,
                       A=A_single, Q=Q_single,
                       Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,B=B,c=c,alpha=alpha,
                       gamma=gamma, state_train=state_train
                       ).params

        _, _, Sensor_Positions_v, Sensor_Chis_v = state_multiple_update_parallel(ps,
                                                                                            U_v,
                                                                                            chis, time_step_sizes)
        
        
        ##target MPC
        U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_velocity, maxval=max_velocity)
        U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_angle_velocity,maxval=max_angle_velocity)
        U_acceleration = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)
        # U_acceleration = jax.random.uniform(key, shape=(M, time_steps, 3 ), minval=min_acceleration, maxval=max_acceleration)
        U_target = U_acceleration
         
        U_target = lbfgsb_target.run(U_target,bounds=bounds, ps_trajectory=Sensor_Positions, pt=pt,chit=chit,
                        time_step_sizes=time_step_sizes,
                        Js=Js, paretos=paretos,
                        A=A_single, Q=Q_single,
                        Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,B=B,c=c,alpha=alpha,
                        gamma=gamma,
                        state_train=state_train).params
         
         #pt=qs[:,:3]
         #vt=qs[:,3:]
        _, _, Target_Positions, Target_Chis = state_target_multiple_update_parallel(pt,U_target,chit,time_step_sizes)
        
            
            
        ps=Sensor_Positions[:,1]
        chis=Sensor_Chis[:,1]
        pt=Target_Positions[:,1]
        chit=Target_Chis[:,1]

        target_states=jnp.concatenate((pt,chit),axis=1)
        sign_vel=np.sign(pt-qs_previous[:,:2])
        A_single=A_single.at[3,3].set(1* sign_vel[0][0])
        A_single=A_single.at[4,4].set(1* sign_vel[0][1])
        
        qs=qs.at[:,:2].set(pt)
        m0=qs
        
 

        Js = [JU_FIM_D_Radar(ps=ps, q=m0[[i],:],U=0, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],B=B,c=c,alpha=alpha) for i in range(len(Js))]
       # Jt=JU_FIM_D_Radar(ps=ps[0,:], q=m0, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[0],B=B,c=c,alpha=alpha)
        traj=np.concatenate((ps.flatten(),qs[:,:3].flatten(),U[:,0,:].flatten()),axis=0)
        traj_list.append(traj)
        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        #J_list.append([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        end = time()
        print(f"Step {k} Optimization Time: ",end-start)
    

    
        Js = [JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],B=B,c=c,alpha=alpha,U=None) for i in range(len(Js))]
    
        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        J_list.append([jnp.log(Ji)[1].item() for Ji in Js])
        print("FIM (higher is better) ",np.sum(J_list[-1]))
    
        save_time = time()
        if (k+1)%frame_skip == 0:
            plt.minorticks_off()
            axes[0].plot(qs_init[:,0], qs_init[:,1], 'b.',label="target_init")
            axes[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="_nolegend_")
            axes[0].plot(m0[:,0], m0[:,1], 'go',label="Targets")
            axes[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")
            axes[0].plot(ps[:,0], ps[:,1], 'rx',label="Sensors")
            axes[0].plot(Sensor_Positions[:,1:,0].T, Sensor_Positions[:,1:,1].T, 'r.-',label="_nolegend_")
            axes[0].plot([],[],"r.-",label="Sensor Planned Path")
            axes[0].plot(Target_Positions[:,1:,0].T, Target_Positions[:,1:,1].T, 'g.-',label="_nolegend_")
            axes[0].plot([],[],"g.-",label="Target Planned Path")
    
            axes[0].legend(bbox_to_anchor=(0.7, 0.95))
            axes[0].set_title(f"k={k}")
    
            qx,qy,logdet_grid = FIM_Visualization(ps=ps, qs=m0,
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s,
                                                  N=250)
    
            axes[1].contourf(qx, qy, logdet_grid, levels=20)
            axes[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes[1].set_title("Instant Time Objective Function Map")
    
            axes[2].plot(jnp.sum(jnp.array(J_list),axis=1))
            axes[2].set_ylabel("sum of individual target logdet FIM")
            axes[2].set_xlabel("Time Step")
    
    
            filename = f"frame_NCIRL_{k}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(photo_dump, filename))
    
            frame_names.append(os.path.join(photo_dump, filename))
            if k <(NT-1):
                axes[0].cla()
                axes[1].cla()
                axes[2].cla()
        save_time = time() - save_time
        print("Figure Save Time: ",save_time)
    
    plt.figure()
    plt.plot(jnp.array(J_list))
    plt.show()
    print("lol")
    
 
    
    
    
    return traj_list ,key


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Optimal Radar Placement', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # =========================== Experiment Choice ================== #
    parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
    parser.add_argument('--frame_skip',default=4,type=int, help='Save the images at every nth frame (must be a multiple of the update on the control frequency, which is dt control / dt ckf)')
    parser.add_argument('--dt_ckf', default=0.1,type=float, help='Frequency at which the radar receives measurements and updated Cubature Kalman Filter')
    parser.add_argument('--dt_control', default=0.1,type=float,help='Frequency at which the control optimization problem occurs with MPPI')
    parser.add_argument('--N_radar',default=1,type=int,help="The number of radars in the experiment")
    parser.add_argument("--N_steps",default=1000,type=int,help="The number of steps in the experiment. Total real time duration of experiment is N_steps x dt_ckf")
    parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
    parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
    parser.add_argument('--move_radars', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to allow the radars to move? --move_radars for yes --no-move_radars for no')
    parser.add_argument('--remove_tmp_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to remove tmp images? --remove_tmp_images for yes --no-remove_tmp_images for no')
    parser.add_argument('--tail_length',default=10,type=int,help="The length of the tail of the radar trajectories in plottings")
    parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')
    parser.add_argument('--fim_method', default="SFIM",type=str, help='FIM Calculation [SFIM,PFIM]')

    # ==================== RADAR CONFIGURATION ======================== #
    parser.add_argument('--fc', default=1e8,type=float, help='Radar Signal Carrier Frequency (Hz)')
    parser.add_argument('--Gt', default=200,type=float, help='Radar Transmit Gain')
    parser.add_argument('--Gr', default=200,type=float, help='Radar Receive Gain')
    parser.add_argument('--rcs', default=1,type=float, help='Radar Cross Section in m^2')
    parser.add_argument('--L', default=1,type=float, help='Radar Loss')
    parser.add_argument('--R', default=500,type=float, help='Radius for specific SNR (desired)')
    parser.add_argument('--Pt', default=1000,type=float, help='Radar Power Transmitted (W)')
    parser.add_argument('--SNR', default=-20,type=float, help='Signal-Noise Ratio for Experiment. Radar has SNR at range R')


    # ==================== MPPI CONFIGURATION ======================== #
    parser.add_argument('--acc_std', default=25,type=float, help='Radar Signal Carrier Frequency (Hz)')
    parser.add_argument('--ang_acc_std', default=45*jnp.pi/180,type=float, help='Radar Transmit Gain')
    parser.add_argument('--horizon', default=15,type=int, help='Radar Receive Gain')
    parser.add_argument('--acc_init', default=0,type=float, help='Radar Cross Section in m^2')
    parser.add_argument('--ang_acc_init', default= 0 * jnp.pi/180,type=float, help='Radar Loss')
    parser.add_argument('--num_traj', default=250,type=int, help='Number of MPPI control sequences samples to generate')
    parser.add_argument('--MPPI_iterations', default=25,type=int, help='Number of MPPI sub iterations (proposal adaptations)')

    # ==================== AIS  CONFIGURATION ======================== #
    parser.add_argument('--temperature', default=0.1,type=float, help='Temperature on the objective function. Lower temperature accentuates the differences between scores in MPPI')
    parser.add_argument('--elite_threshold', default=0.9,type=float, help='Elite Threshold (between 0-1, where closer to 1 means reject most samaples)')
    parser.add_argument('--AIS_method', default="CE",type=str, help='Type of importance sampling. [CE,information]')

    # ============================ MPC Settings =====================================#
    parser.add_argument('--gamma', default=0.95,type=float, help="Discount Factor for MPC objective")
    parser.add_argument('--speed_minimum', default=5,type=float, help='Minimum speed Radars should move [m/s]')
    parser.add_argument('--R2T', default=125,type=float, help='Radius from Radar to Target to maintain [m]')
    parser.add_argument('--R2R', default=10,type=float, help='Radius from Radar to  Radar to maintain [m]')
    parser.add_argument('--alpha1', default=1,type=float, help='Cost weighting for FIM')
    parser.add_argument('--alpha2', default=1000,type=float, help='Cost weighting for maintaining distanace between Radar to Target')
    parser.add_argument('--alpha3', default=60,type=float, help='Cost weighting for maintaining distance between Radar to Radar')
    parser.add_argument('--alpha4', default=1,type=float, help='Cost weighting for smooth controls (between 0 to 1, where closer to 1 means no smoothness')
    parser.add_argument('--alpha5', default=0,type=float, help='Cost weighting to maintain minimum absolute speed')


    args = parser.parse_args()


    args.results_savepath = os.path.join(args.results_savepath,args.experiment_name) + f"_{args.seed}"
    args.tmp_img_savepath = os.path.join( args.results_savepath,"tmp_img") #('--tmp_img_savepath', default=os.path.join("results","tmp_images"),type=str, help='Folder to save temporary images to make GIFs')



    from datetime import datetime
    from pytz import timezone
    import json

    tz = timezone('EST')
    print("Experiment State @ ",datetime.now(tz))
    print("Experiment Saved @ ",args.results_savepath)
    print("Experiment Settings Saved @ ",args.results_savepath)


    os.makedirs(args.tmp_img_savepath,exist_ok=True)
    os.makedirs(args.results_savepath,exist_ok=True)

    # Convert and write JSON object to file
    with open(os.path.join(args.results_savepath,"hyperparameters.json"), "w") as outfile:
        json.dump(vars(args), outfile)

    N= 1
    M=1
    dn=2
    dm=6
    state_shape =((M*(dm//2) + N*(dn+1),))
    input_shape=state_shape
    cost_f = CostNN(state_dims=state_shape[0])
    #cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
    init_rng = jax.random.key(0)

    variables = cost_f.init(init_rng, jnp.ones((1,input_shape[0]))) 

    params = variables['params']
    tx = optax.adam(learning_rate=1e-2)
    state_train=train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
    
    #traj_list=generate_demo(state_train)
    traj_list,key=generate_demo_MPPI_single(args,state_train=None)
    

    
    scipy.io.savemat('Single_traj_follow.mat', {'traj': traj_list})

    



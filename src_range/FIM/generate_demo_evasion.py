# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:12:42 2024

@author: siliconsynapse
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:58:22 2024

@author: siliconsynapse
"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ScipyMinimize,ScipyBoundedMinimize
import numpy as np
import scipy 
import imageio
import matplotlib
#matplotlib.use('Agg')
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tqdm import tqdm
from time import time
from copy import deepcopy

import os
import glob

from src_range.FIM.JU_Radar import *
from src_range.utils import NoiseParams
from src_range.FIM_new.FIM_RADAR import FIM_Visualization
from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar  as Single_JU_FIM_Radar_MPPI
from src_range.FIM_new.FIM_RADAR import Multi_FIM_Logdet_decorator_MPC as Multi_FIM_Logdet_decorator_MPPI
from src_range.control.MPPI import *

from cost_jax import CostNN, apply_model, update_model
from flax.training import train_state,checkpoints
import flax 
import optax

config.update("jax_enable_x64", True)





if __name__ == "__main__":



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
    frames = []
    frame_names = []
    state_multiple_update_parallel = vmap(state_multiple_update, (0, 0, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_velocity+5, maxval=max_velocity)
    U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=min_angle_velocity,maxval=max_angle_velocity)
    U = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)
    
    method_t = "Single_FIM_3D_evasion_MPPI"
    #MPPI_method = "single"
    MPPI_method_t="single"
    IM_method="MPPI_method_evasion"
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

    IM_fn_t = partial(Single_JU_FIM_Radar_MPPI,actions=None,state_train=None,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW,method=IM_method)
    #IM_fn(ps,qs,J=J,actions=U_Nom)
    IM_fn_GT = partial(Single_JU_FIM_Radar_MPPI,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW)


    Multi_FIM_Logdet_t = Multi_FIM_Logdet_decorator_MPPI(IM_fn=IM_fn_t,method=method_t)

    MPPI_scores_t = MPPI_scores_wrapper(Multi_FIM_Logdet_t,method=MPPI_method_t)
    
    J = jnp.eye(dm*M)
    target_states=jnp.concatenate((pt,chit),axis=1)
    state_shape =((M*(dm//2) + N*(dn+1),))
    input_shape=state_shape
    cost_f = CostNN(state_dims=state_shape[0])
    #cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
    init_rng = jax.random.key(0)

    variables = cost_f.init(init_rng, jnp.ones((1,input_shape[0]))) 

    params = variables['params']
    tx = optax.adam(learning_rate=1e-2)
    state_train=train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
    state_train=None
    m0=m0.at[:,:2].set(pt)
    state_multiple_update_vmap = vmap(state_multiple_update, (0, 0, 0, None))
    
    for k in range(NT):
        start = time()
        m0=m0.at[:,:2].set(pt)
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
  
        
        # MPPI 
        best_mppi_iter_score = -np.inf
        best_mppi_iter_score_t = -np.inf
        mppi_round_time_start = time()
        
    
        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])

        
        
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
                                      gamma=gamma)
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
        ps = Sensor_Positions[:,1,:]
        chis = Sensor_Chis[:,1,:]
        pt=Target_Positions[:,1]
        chit=Target_Chis[:,1]
        target_states=jnp.concatenate((pt,chit),axis=1)
        sign_vel=np.sign(pt-qs_previous[:,:2])
        A_single=A_single.at[3,3].set(1* sign_vel[0][0])
        A_single=A_single.at[4,4].set(1* sign_vel[0][1])
        
        
        
        

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


            filename = f"frame_{k}.png"
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
    
    scipy.io.savemat('Single_traj_follow.mat', {'traj': traj_list})

    



# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:06:20 2024

@author: siliconsynapse
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:21:43 2024

@author: siliconsynapse
"""

import numpy as np

from scipy.optimize import minimize


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
import jax
from src.FIM.JU_Radar import *
from src.utils import NoiseParams


config.update("jax_enable_x64", True)

if __name__ == "__main__":



    seed = 555
    

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"

    # Save frames as a GIF
    gif_filename = "radar_optimal_RICE.gif"
    gif_savepath = os.path.join("gifs")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = .05
    NT = 300
    N = 5

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    alpha = (np.pi)**2 / 3
    B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    K = Gt * Gr * lam ** 2 * rcs / L / (4 * np.pi) ** 3
    coef = K / (R ** 4)


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
    max_acceleration = 5.
    min_acceleration = -5.
    max_angle_velocity = np.pi/2
    min_angle_velocity = -np.pi/2

    # ==================== MPPI CONFIGURATION ================================= #
    limits = np.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])

    # ps = place_sensors([-100,100],[-100,100],N)
    
    #
    ps = np.random.uniform(-100, 100,(N, 2))
    ps_init = deepcopy(ps)
    z_elevation=10
    qs = np.array([[0.0, -0.0, z_elevation,25., 20,0], #,#,
                    [-50.4,30.32, z_elevation,-20,-10,0], #,
                    [10,10, z_elevation,10,10,0],
                    [20,20, z_elevation,5,-5,0]])
    # qs = jnp.array([[0.0, -0.0, 25., 20], #,#,
    #                 [-50.4,30.32,-20,-10], #,
    #                 [10,10,10,10],
    #                 [20,20,5,-5]])
    qs_init = deepcopy(qs)
    M, d = qs.shape;
    N = len(ps);

    # ======================== MPC Assumptions ====================================== #
    gamma = 0.9
    paretos = np.ones((M,)) * 1 / M  # jnp.array([1/3,1/3,1/3])
    
    paretos_t = np.array([1,0,0,0])

    assert len(paretos) == M, "Pareto weights not equal to number of targets!"
    assert len(paretos_t) == M, "Pareto weights not equal to number of targets!"
    assert (np.sum(paretos) <= (1 + 1e-5)) and (np.sum(paretos) >= -1e-5), "Pareto weights don't sum to 1!"


    sigmaQ = np.sqrt(10 ** 2);

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

    A_single = np.array([[1., 0, 0, T, 0, 0],
                   [0, 1., 0, 0, T, 0],
                   [0, 0, 1, 0, 0, T],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = np.array([
        [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
        [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
        [0, 0, (T**4)/4, 0, 0, (T**3) / 2],
        [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
        [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
        [0, 0, (T**3) / 2, 0, 0, (T**2)]
    ]) * sigmaQ ** 2

    A = np.kron(np.eye(M), A_single);
    Q = np.kron(np.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = np.eye(N)

    nx = Q.shape[0]

    Js = [np.eye(d) for m in range(M)]

    JU_FIM_D_Radar(ps, q=qs[[0],:], J=Js[0],
                   A=A_single, Q=Q_single,
                   Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC_scipy(JU_FIM_radareqn_target_logdet)
    Multi_FIM_target_Logdet = Multi_FIM_Logdet_decorator_MPC(JU_FIM_radareqn_target_logdet, method='distraction')
   

    print("Optimization START: ")
    lbfgsb =  ScipyBoundedMinimize(fun=Multi_FIM_Logdet, method="L-BFGS-B", dtype=np.float64, jit=True)
    lbfgsb_target =  ScipyBoundedMinimize(fun=Multi_FIM_target_Logdet, method="L-BFGS-B",jit=True)

    chis = np.random.uniform(-np.pi,np.pi,(ps.shape[0],1)) #jnp.tile(0., (ps.shape[0], 1, 1))
    time_step_sizes = np.tile(time_step_size, (N, 1))
    time_step_sizes_target = np.tile(time_step_size, (M, 1))

    U_upper = (np.ones((time_steps, 2)) * np.array([[max_velocity, max_angle_velocity]]))
    U_lower = (np.ones((time_steps, 2)) * np.array([[min_velocity, min_angle_velocity]]))

    U_lower = np.tile(U_lower, np.array([N, 1, 1]))
    U_upper = np.tile(U_upper, np.array([N, 1, 1]))
    bounds = (U_lower, U_upper)
    
    U_target_upper = (np.ones((time_steps, 3)) * np.array([[max_acceleration]]))
    U_target_lower = (np.ones((time_steps, 3)) * np.array([[min_acceleration]]))

    U_target_lower = np.tile(U_target_lower, np.array([M, 1, 1]))
    U_target_upper = np.tile(U_target_upper, np.array([M, 1, 1]))
    bounds_target = (U_target_lower, U_target_upper)
    
    bounds=[]
    
    
    for i in range (time_steps*N):
        bounds+=[[min_velocity, max_velocity]]
       
    
        
    for i in range (time_steps*N):
        bounds+=[[min_angle_velocity, max_angle_velocity]]
    
    m0 = qs

    J_list = []
    frames = []
    frame_names = []
    state_multiple_update_parallel = vmap(state_multiple_update, (0, 0, 0, 0))
    state_target_multiple_update_parallel = vmap(state_target_multiple_update, (0, 0, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        
   

    for k in range(NT):
        start = time()
        qs_previous = qs

        m0 = (A @ m0.reshape(-1, 1)).reshape(M, d)

        U_velocity = np.random.uniform(min_velocity, max_velocity,(N, time_steps, 1 ))
        U_angular_velocity = np.random.uniform(min_angle_velocity, max_angle_velocity,(N, time_steps, 1 ))
        U_velocity=U_velocity.flatten()
        U_angular_velocity=U_angular_velocity.flatten()
         
        U = np.concatenate((U_velocity, U_angular_velocity), axis=-1)
        
        objective= lambda U: Multi_FIM_Logdet(U=U, chis=chis, ps=ps, qs=m0,
                       time_step_sizes=time_step_sizes,
                       Js=Js, paretos=paretos,
                       A=A_single, Q=Q_single,
                       Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,B=B,c=c,alpha=alpha,
                       gamma=gamma,
                       horizon=time_steps)
        
        jac_jax = jax.jit(jax.grad(objective,argnums=0))
        hess_jax = jax.jit(jax.hessian(objective,argnums=0))

        #jac = lambda ps_optim: jac_jax(ps_optim).ravel()
        #hess = lambda ps_optim: hess_jax(ps_optim)
        # U = jnp.zeros((N,2,time_steps))
        
        U = minimize(objective,
                              x0=U,
                              method='L-BFGS-B',
                              bounds=bounds,jac=jac_jax,
                              options={'maxiter':1000 ,'disp':True}).x
        
        U_velocity=U[0:50]
        U_velocity=U_velocity.reshape((N,time_steps,1))
        U_angular_velocity=U[50:]
        U_angular_velocity=U_angular_velocity.reshape((N,time_steps,1))
        U=np.concatenate((U_velocity, U_angular_velocity), axis=-1)
        

        _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_parallel(ps,
                                                                                            U,
                                                                                            chis, time_step_sizes)
        ps = Sensor_Positions[:,1,:]
        chis = Sensor_Chis[:,1,:]


        # print(ps.shape,chis.shape,ps.squeeze().shape)
        # ps = ps.squeeze()
        # chis = chis.squeeze()


        end = time()
        print(f"Step {k} Optimization Time: ",end-start)

        # m0  = ps

        Js = [JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],B=B,c=c,alpha=alpha) for i in range(len(Js))]

        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        J_list.append([np.linalg.slogdet(Ji)[1].item() for Ji in Js])
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
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s,
                                                  N=250)

            axes[1].contourf(qx, qy, logdet_grid, levels=20)
            axes[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes[1].set_title("Instant Time Objective Function Map")

            axes[2].plot(np.sum(np.array(J_list),axis=1))
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

    plt.figure()
    plt.plot(np.array(J_list))
    plt.show()
    print("lol")


    for frame_name in frame_names:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, gif_filename), frames, duration=0.25)
    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump, "frame_*")):
            os.remove(filename)



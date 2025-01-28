# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:38:21 2024

@author: siliconsynapse
"""



import jax
import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import scipy.io as sio
from copy import deepcopy
from jax import config
from flax.training import train_state,checkpoints
import flax 
import optax
import argparse
from time import time
import os
from tqdm import tqdm





from experts.P_MPPI import *
from cost_jax import CostNN, apply_model, update_model,get_gradients,get_hessian
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR
from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,JU_RANGE_SFIM,Single_FIM_Radar,FIM_Visualization
from src_range.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range.utils import visualize_tracking,visualize_control,visualize_target_mse,place_sensors_restricted
from src_range.control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper #,MPPI_adapt_distribution
from src_range.objective_fns.objectives import *
from src_range.tracking.cubatureTestMLP import generate_data_state
from src_range.FIM_new.generate_demo_fn import generate_demo_MPPI_single

import imageio

import shutil



#torch.autograd.set_detect_anomaly(True)
# SEEDS


# ENV SETUP

parser = argparse.ArgumentParser(description = 'Optimal Radar Placement', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# =========================== Experiment Choice ================== #
parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
parser.add_argument('--frame_skip',default=4,type=int, help='Save the images at every nth frame (must be a multiple of the update on the control frequency, which is dt control / dt ckf)')
parser.add_argument('--dt_ckf', default=0.1,type=float, help='Frequency at which the radar receives measurements and updated Cubature Kalman Filter')
parser.add_argument('--dt_control', default=0.1,type=float,help='Frequency at which the control optimization problem occurs with MPPI')
parser.add_argument('--N_radar',default=1,type=int,help="The number of radars in the experiment")
parser.add_argument("--N_steps",default=200,type=int,help="The number of steps in the experiment. Total real time duration of experiment is N_steps x dt_ckf")
parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
parser.add_argument('--move_radars', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to allow the radars to move? --move_radars for yes --no-move_radars for no')
parser.add_argument('--remove_tmp_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to remove tmp images? --remove_tmp_images for yes --no-remove_tmp_images for no')
parser.add_argument('--tail_length',default=10,type=int,help="The length of the tail of the radar trajectories in plottings")
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=False,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')
parser.add_argument('--fim_method_policy', default="SFIM_NN",type=str, help='FIM Calculation [SFIM,PFIM]')
parser.add_argument('--fim_method_demo', default="SFIM",type=str, help='FIM Calculation [SFIM,PFIM]')

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
 
# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = jnp.array(traj[0])
        if is_Demo:
            probs = jnp.ones((states.shape[0], 1))
        else:
            probs = jnp.array(traj[1]).reshape(-1, 1)
        actions = jnp.array(traj[2])
        x = jnp.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return jnp.array(step_list)   
    
def cost_fn(state_train,params,states,N):
    distances=states[:,:3]-states[:,3:]
    #costs_demo = -jnp.log(state_train.apply_fn({'params': params}, states_expert)+1e-2)
    #costs_samp =-jnp.log(state_train.apply_fn({'params': params}, states)+1e-2)
    costs = -jnp.log(state_train.apply_fn({'params': params}, distances)+1e-6).flatten()/N
    
    return costs[0].astype(float)




n_actions = 2
M,dm=1,6
N,dn=1,2
#state_shape =((M*(dm//2) + N*(dn+1),))
state_shape =((M*(dm//2),))





# INITILIZING POLICY AND REWARD FUNCTION
policy = P_MPPI(state_shape, n_actions)
cost_f = CostNN(state_dims=state_shape[0])
#cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
init_rng = jax.random.key(0)

variables = cost_f.init(init_rng, jnp.ones((1,state_shape[0]))) 

params = variables['params']
#params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
#params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
tx = optax.adam(learning_rate=1e-3)
state_train=train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
theta=jnp.concatenate((params['Dense_0']['bias'].flatten(),params['Dense_0']['kernel'].flatten(),params['Dense_1']['bias'].flatten(),params['Dense_1']['kernel'].flatten()))
n_theta=len(theta)
P_theta=1e-1*jnp.identity(n_theta)
Q_theta=1e-3*jnp.identity(n_theta)

#dc_d_theta=jax.grad(cost_f,argnums=1)(state_train,params,states)
mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 500
sample_trajs = []

D_demo, D_samp = np.array([]), jnp.array([])


#D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
#D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
return_list, sum_of_cost_list = [], []

#U,chis,radar_states,target_states
thetas=jnp.ones((1,4))
mpc_method = "Single_FIM_3D_action_NN_MPPI"
FIM_true=[]
FIM_predicted=[]
epoch_time=[]
for i in range(50):
    
    if (i %10 ==0 and i >0) :
         Q_theta=1e-1*Q_theta
    if (i== 0): 
    
        demo_trajs,demo_trajs_sindy,FIM_demo=generate_demo_MPPI_single(args,state_train=None)
        demo_trajs=np.array(demo_trajs)
    
        states_d=demo_trajs[:,:-2]
        actions_d=demo_trajs[:,-2:]
        D_demo=np.array([])
    
        demo_trajs=[[states_d,actions_d,actions_d]]
        D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
        D_demo=jnp.concatenate((D_demo[:,:2],D_demo[:,2:]),axis=1)
        
    start_time=time()   
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

    #print("Noise Power: ",sigmaW**2)
    #print("Power Return (RCS): ",Pr)
    #print("K",K)

    #print("Pt (peak power)={:.9f}".format(args.Pt))
    #print("lam ={:.9f}".format(c/args.fc))
    #print("C=",C)

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
    
    J = jnp.linalg.inv(P)
    J_NN=J
    #trajs = [policy.generate_session(args,i,state_train,D_demo,mpc_method)]
    
    
    
    
   
    #sample_trajs = trajs #+ sample_trajs
    #sample_trajs = demo_trajs + sample_trajs
    #D_samp=np.array([])
    #D_samp = preprocess_traj(trajs, D_samp)
    #D_s_samp = D_samp
    D_s_demo = D_demo
    #states, probs, actions = D_s_samp[:,:-3], D_s_samp[:,-3], D_s_samp[:,-2:]
    states_expert,probs_experts, actions_expert = D_s_demo[:,:-3], D_s_demo[:,-3], D_s_demo[:,-2:]
    traj=[]
    MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method,state_train=state_train,thetas=thetas)
    #MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method)
    MPPI_scores = MPPI_scores_wrapper(MPC_obj,method="NN")
    pbar = tqdm(total=args.N_steps, desc="Starting")

    for step in range(1,args.N_steps+1):
        


        #MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)

        target_state_true = target_states_true[:, step-1].reshape(M_target,dm)
        #radar_state=states_expert[step-1].reshape(1,-1)
        demo_x_min,demo_x_max= jnp.min(D_demo[:,0]),jnp.max(D_demo[:,0])
        demo_y_min,demo_y_max= jnp.min(D_demo[:,1]),jnp.max(D_demo[:,1])
        
        
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
                
                # radar_states=radar_states.at[:,:,0].set(jnp.clip(radar_states[:,:,0],demo_x_min,demo_x_max))
                # radar_states=radar_states.at[:,:,1].set(jnp.clip(radar_states[:,:,1],demo_y_min,demo_y_max))
                # radar_states_MPPI=radar_states_MPPI.at[:,:,:,0].set(jnp.clip(radar_states_MPPI[:,:,:,0],demo_x_min,demo_x_max))
                # radar_states_MPPI=radar_states_MPPI.at[:,:,:,1].set(jnp.clip(radar_states_MPPI[:,:,:,1],demo_y_min,demo_y_max))
                

                mppi_rollout_end = time()
                

                # GET MPC OBJECTIVE
                mppi_score_start = time()
                # Score all the rollouts
                cost_trajectory = MPPI_scores(radar_state, target_states_rollout, V,
                                          A=A,J=J,state_train=state_train)
                

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
                    #if mppi_iter == 0 and i%10 ==0:
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
            
            #print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))



        
        
        state=jnp.concatenate((radar_state[:,:3].flatten(),target_state_true[:,0:3].flatten()))
        #traj.append(state)
        #selected_samp = traj[np.random.choice(len(traj), 1)]
        gradient_s=get_gradients(state_train,params,state,args.N_steps)
        gradient_d=get_gradients(state_train,params,states_expert[step-1],args.N_steps)
        hessian_s=get_hessian(state_train,params,state,args.N_steps)
        hessian_d=get_hessian(state_train,params,states_expert[step-1],args.N_steps)
        

        P_theta=jnp.linalg.inv(jnp.linalg.inv(P_theta +Q_theta) + hessian_d - hessian_s)
        #print(gradient_d)
        #print(gradient_s)
       
        theta=theta-jnp.matmul(P_theta,gradient_d-gradient_s)
        
        params['Dense_0']['bias']=theta[:len(params['Dense_0']['bias'])]
        params['Dense_0']['kernel']=theta[len(params['Dense_0']['bias']):len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]].reshape(params['Dense_0']['kernel'].shape)
        params['Dense_1']['bias']=theta[len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]:len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]+len(params['Dense_1']['bias'])]
        params['Dense_1']['kernel']=theta[len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]+len(params['Dense_1']['bias']):].reshape(-1,1)
        
        J_NN = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                  J=J,method="NN",state_train=state_train,thetas=thetas)
        J = IM_fn_update(radar_state=radar_state, target_state=target_state_true,
                  J=J)

        # print(jnp.linalg.slogdet(J)[1].ravel().item())
        if args.N_radar==1 and M_target==1:
            FIMs_NN[step] = jnp.log(J_NN).ravel().item()
            FIMs[step]=jnp.log(J).ravel().item()
        else:
            
            FIMs_NN[step] = jnp.linalg.slogdet(J_NN)[1].ravel().item()
            FIMs[step] = jnp.linalg.slogdet(J)[1].ravel().item()

        radar_state_history[step] = radar_state
       
        #print("FIM :" , FIMs[step])
        #print("Thetas :" , thetas)
        pbar.set_description(
                f"FIM = {FIMs[step]} ")
        pbar.update(1)
        
        
        if args.save_images and (step % 4 == 0):
            #print(f"Step {step} - Saving Figure ")

            axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                     label="Radar Initial Position")

            thetas_1 = jnp.arcsin(target_state_true[:, 2] / args.R2T)
            radius_projected = args.R2T * jnp.cos(thetas_1)
#
            #print("Target Height :",target_state_true[:,2])
            #print("Radius Projected: ",radius_projected)

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
    pbar.close()
    end_time=time()
    epoch_time.append(end_time-start_time)
    FIM_true.append(FIMs)
    FIM_predicted.append(FIMs_NN)
    np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')

    if args.save_images:
        visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

        images = [imageio.imread(file) for file in imgs_main]
        imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method_policy}.gif'), images, duration=0.1)

        images = [imageio.imread(file) for file in imgs_control]
        imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


        if args.remove_tmp_images:
            shutil.rmtree(args.tmp_img_savepath)
   
    
# config = {'dimensions': np.array([5, 3])}
# ckpt_single = {'model_single': state_train, 'config': config, 'data': [D_samp]}
# checkpoints.save_checkpoint(ckpt_dir='/tmp/flax_ckpt/flax-checkpointing',
#                             target=ckpt_single,
#                             step=0,
#                             overwrite=True,
#                             keep=2)
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:59:39 2024

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
from jax import vmap





from experts.P_MPPI import *
from cost_jax import CostNN, apply_model, update_model
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR
from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,JU_RANGE_SFIM,Single_FIM_Radar,FIM_Visualization,features_f
from src_range.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range.utils import visualize_tracking,visualize_control,visualize_target_mse,place_sensors_restricted
from src_range.control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper #,MPPI_adapt_distribution
from src_range.objective_fns.objectives import *
from src_range.tracking.cubatureTestMLP import generate_data_state
from src_range.FIM_new.generate_demo_fn import generate_demo_MPPI_single


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

def features_f_tilde(radar_state,target_state,thetas,mean):
    radar_state=radar_state.reshape((1,-1))
    target_state=target_state.reshape((1,-1))
    
    features_tilde=jnp.exp(-jnp.matmul(thetas,features_f(radar_state,target_state)))*features_f(radar_state,target_state)
   # features_tilde=jnp.exp(-jnp.matmul(thetas,features_f(radar_state,target_state)))
    
    return features_tilde/mean

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
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')
parser.add_argument('--fim_method_policy', default="SFIM_features",type=str, help='FIM Calculation [SFIM,PFIM]')
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


n_actions = 2
M,dm=1,6
N,dn=1,2
state_shape =((M*(dm//2) + N*(dn+1),))





# INITILIZING POLICY AND REWARD FUNCTION
policy = P_MPPI(state_shape, n_actions)
cost_f = CostNN(state_dims=state_shape[0])
#cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
init_rng = jax.random.key(0)

variables = cost_f.init(init_rng, jnp.ones((1,state_shape[0]))) 

params = variables['params']
#params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
#params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
tx = optax.adam(learning_rate=1e-2)
state_train=train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)

mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), jnp.array([])


#D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
#D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
return_list, sum_of_cost_list = [], []

#U,chis,radar_states,target_states
thetas=1e-3*jnp.ones((1,6))

alpha =1e-3
mpc_method = "Single_FIM_3D_action_features_MPPI"
features_func=vmap(features_f,in_axes=(0,0))
features_tilde_func=vmap(features_f_tilde,in_axes=(0,0,None,None))

for i in range(100):
    #D_samp=jnp.array([])
    if (i== 0): 
    
        demo_trajs,demo_trajs_sindy=generate_demo_MPPI_single(args,state_train=None)
        demo_trajs=np.array(demo_trajs)
    
        states_d=demo_trajs[:,:-2]
        actions_d=demo_trajs[:,-2:]
        D_demo=np.array([])
    
        demo_trajs=[[states_d,actions_d,actions_d]]
        D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
        D_demo=jnp.concatenate((D_demo[:,:2],D_demo[:,2:]),axis=1)
    
    trajs= [policy.generate_session(args,i,actions_d,state_train,D_demo,mpc_method,thetas)]
    trajs_sindy=trajs[0][3]
    trajs=[trajs[0][:3]]
    
    #sample_trajs = sample_trajs
    #sample_trajs = demo_trajs + sample_trajs
    D_samp = preprocess_traj(trajs, D_samp)
    #D_samp = D_demo
    
    states, probs, actions = D_samp[:,:-3], D_samp[:,-3], D_samp[:,-2:]
    states_expert,probs_experts, actions_expert = D_demo[:,:-3], D_demo[:,-3], D_demo[:,-2:]
    
    S_exp,ds=states_expert.shape
    S,ds=states.shape
    radar_experts=states_expert[:,:3].reshape((S_exp,3))
    target_experts=states_expert[:,3:].reshape((S_exp,3))
    radar_samples=states[:,:3].reshape((S,3))
    target_samples=states[:,3:].reshape((S,3))
    
    
    
    
    mean=jnp.mean(jnp.exp(-jnp.matmul(thetas,features_func(radar_samples,target_samples))),axis=0)
    
    gradients=-jnp.mean(features_func(radar_experts,target_experts),axis=0)+ jnp.mean(features_tilde_func(radar_samples,target_samples,thetas,mean),axis=0)
    #gradients= - jnp.mean(features_tilde_func(radar_samples,target_samples,thetas,mean),axis=0)
   
    thetas= thetas -alpha*gradients.T
    print(thetas)
    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    

    

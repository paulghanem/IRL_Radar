# %%
from flax.training import train_state,checkpoints

import flax 
import optax
import os
import argparse
import os.path as osp

import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap,jit
import time 


import gymnax
import gymnasium as gym
import mujoco
from gymnax.visualize import Visualizer
from flax import struct
from gymnax.environments import EnvState
import pdb
from mujoco import mjx 

# from experts.P_MPPI import P_MPPI
from cost_jax import CostNN, apply_model, apply_model_AIRL,update_model,apply_model_SQIL

from src.objective_fns.cost_to_go_fns import get_cost
from src.control.dynamics import get_state
from src.control.mppi_class import MPPI
from src.control.PPO import PPOPolicy,policy_model
from src.control.dynamics import get_action_cov,get_action_space,get_step_model

from utils.helpers import GenerateDemo

import gymnax

print(jax.devices())
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


#torch.autograd.set_detect_anomaly(True)
# SEEDS


# ENV SETUP

parser = argparse.ArgumentParser(description = 'Optimal Radar Placement', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# =========================== Experiment Choice ================== #
parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
parser.add_argument("--N_steps_expert",default=1000000,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--N_steps",default=2000,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--rirl_iterations",default=1,type=int,help="The number of epoch updates")
parser.add_argument("--reward_fn_updates",default=10,type=int,help="The number of reward fn updates")
parser.add_argument("--hidden_dim",default=16,type=int,help="The number of hidden neurons")
parser.add_argument("--lambda_",default=0.01,type=float,help="Temperature in MPPI (lower makers sharper)")
parser.add_argument("--runs",default=10,type=int,help="The number of runs")

parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')


parser.add_argument('--lr', default=1e-4,type=float, help='learning rate')
parser.add_argument('--P', default=1e-2,type=float, help='rgcl initial covariance')
parser.add_argument('--Q', default=1e-5,type=float, help='rgcl learning rate')
parser.add_argument('--sigma', default=0.0,type=float, help='noise level')

parser.add_argument("--UB",action=argparse.BooleanOptionalAction,default=False,type=bool,help="Upper bound loss  ")
parser.add_argument('--sqil', action=argparse.BooleanOptionalAction,default=False,type=bool, help='sqil method flag (automatically turns sqil flag on)')
parser.add_argument('--gail', action=argparse.BooleanOptionalAction,default=False,type=bool, help='gail method flag (automatically turns gail flag on)')
# %%
parser.add_argument('--airl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='airl method flag')

parser.add_argument('--rgcl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='rgcl method flag')
parser.add_argument('--gym_env', default="HalfCheetah-v4",type=str, help='gym environment to test (CartPole-v1 , Pendulum-v1)')
parser.add_argument('--PPO', action=argparse.BooleanOptionalAction,default=False,type=bool, help='PPO policy flag')

parser.add_argument("--online",action=argparse.BooleanOptionalAction,default=False,type=bool,help="online version of bechmarks ")

parser.add_argument("--diagonal",action=argparse.BooleanOptionalAction,default=False,type=bool,help="diagonal version of hessians ")

# ==================== MPPI CONFIGURATION ======================== #
parser.add_argument('--horizon', default=20,type=int, help='Horizon for MPPI control')
parser.add_argument('--num_traj', default=2000,type=int, help='Number of MPPI control sequences samples to generate')




args = parser.parse_args()


      
    






# policy_method_agent = "es" if args.gym_env == "MountainCarContinuous-v0" else "ppo"
# base = osp.join("expert_agents", args.gym_env, policy_method_agent)
# configs = load_config(base + ".yaml")
# model, model_params = load_neural_network(
#     configs.train_config, base + ".pkl"
# )
#
# env, env_params = gymnax.make(
#     configs.train_config.env_name,
#     **configs.train_config.env_kwargs,
# )
# env_params.replace(**configs.train_config.env_params)
#
#
# states_d,actions_d,_ =generate_demo(
#     env, env_params, model, model_params,max_frames=DEMO_BATCH,seed=args.seed)
demo_generator = GenerateDemo(args.gym_env,max_frames=args.N_steps_expert)
states_d,actions_d,rewards_demo,env = demo_generator.generate_demo(args.seed)
            
fname = f"expert_demo_{args.gym_env}_seed{args.seed}.npz"
np.savez(fname, states=states_d, actions=actions_d, rewards=rewards_demo,env=env)
       
            
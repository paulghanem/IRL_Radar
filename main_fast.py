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
import sys

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
from src.control.mppi_fast import FastMPPI,generate_session_fast_mppi
from src.control.PPO import PPOPolicy,policy_model
from src.control.dynamics import get_action_cov,get_action_space,get_step_model

from utils.helpers import GenerateDemo

import gymnax
# Redirect NumPy 2.x paths to NumPy 1.x
#sys.modules["numpy._core"] = np.core
#sys.modules["numpy._core.numeric"] = np.core.numeric

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



parser = argparse.ArgumentParser(description = 'Optimal Radar Placement', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# =========================== Experiment Choice ================== #
parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
parser.add_argument("--N_steps_expert",default=2,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--N_steps",default=2,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--rirl_iterations",default=100,type=int,help="The number of epoch updates")
parser.add_argument("--reward_fn_updates",default=15,type=int,help="The number of reward fn updates")
parser.add_argument("--hidden_dim",default=16,type=int,help="The number of hidden neurons")
parser.add_argument("--lambda_",default=0.01,type=float,help="Temperature in MPPI (lower makers sharper)")
parser.add_argument("--runs",default=10,type=int,help="The number of runs")

parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')


parser.add_argument('--lr', default=1e-4,type=float, help='learning rate')
parser.add_argument('--P', default=1e-2,type=float, help='rgcl initial covariance')
parser.add_argument('--Q', default=1e-4,type=float, help='rgcl learning rate')
parser.add_argument('--sigma', default=0.0,type=float, help='noise level')

parser.add_argument("--UB",action=argparse.BooleanOptionalAction,default=False,type=bool,help="Upper bound loss  ")
parser.add_argument('--sqil', action=argparse.BooleanOptionalAction,default=False,type=bool, help='sqil method flag (automatically turns sqil flag on)')
parser.add_argument('--gail', action=argparse.BooleanOptionalAction,default=False,type=bool, help='gail method flag (automatically turns gail flag on)')
# %%
parser.add_argument('--airl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='airl method flag')

parser.add_argument('--rgcl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='rgcl method flag')
parser.add_argument('--gym_env', default="Walker2d",type=str, help='gym environment to test (CartPole-v1 , Pendulum-v1)')
parser.add_argument('--PPO', action=argparse.BooleanOptionalAction,default=False,type=bool, help='PPO policy flag')

parser.add_argument("--online",action=argparse.BooleanOptionalAction,default=False,type=bool,help="online version of bechmarks ")

parser.add_argument("--diagonal",action=argparse.BooleanOptionalAction,default=False,type=bool,help="diagonal version of hessians ")

# ==================== MPPI CONFIGURATION ======================== #
parser.add_argument('--horizon', default=5,type=int, help='Horizon for MPPI control')
parser.add_argument('--num_traj', default=500,type=int, help='Number of MPPI control sequences samples to generate')




args = parser.parse_args()


args.airl = True if args.gail else args.airl
args.airl = False if args.rgcl else args.airl
args.gail = False if args.rgcl else args.gail

print("Using AIRL: ",args.airl)
print("Using GAIL: ",args.gail)
print("Using RGCL: ",args.rgcl)

if args.airl and not args.gail:
    if args.online:
        method="airl-online"
    else:
        method="airl"
elif args.gail:
    if args.online:
        method="gail-online"
    else:
        method="gail"
elif args.rgcl:
    method="rgcl"
    if args.diagonal:
        method="rgcl-diagonal"
elif args.UB:
    if args.online:
        method="UB-online"
    else:
        method="UB"
elif args.sqil:
    if args.online:
        method="sqil-online"
    else:
        method="sqil"
else :
    if args.online:
        method="gcl-online"
    else:
        method="gcl"

      
    

args.results_savepath = os.path.join(args.results_savepath,args.experiment_name) + f"_{args.seed}"
args.tmp_img_savepath = os.path.join( args.results_savepath,"tmp_img") #('--tmp_img_savepath', default=os.path.join("results","tmp_images"),type=str, help='Folder to save temporary images to make GIFs')



from datetime import datetime
from pytz import timezone
import json

tz = timezone('EST')
print("Experiment State @ ",datetime.now(tz))
print("Experiment Saved @ ",args.results_savepath)
print("Experiment Settings Saved @ ",args.results_savepath)

mjx_model=None
os.makedirs(args.tmp_img_savepath,exist_ok=True)
os.makedirs(args.results_savepath,exist_ok=True)

# Convert and write JSON object to file
with open(os.path.join(args.results_savepath,"hyperparameters.json"), "w") as outfile:
    json.dump(vars(args), outfile)

assets_dir="assets"
args.dt=1
args.frame_skip=1
if args.gym_env in ["HalfCheetah-v4","Ant-v4","Hopper","Walker2d","Humanoid-v4","Swimmer"]:
    if args.gym_env=="HalfCheetah-v4":
        env_xml = "half_cheetah.xml"
        args.frame_skip=5
        args.dt=0.01
    elif args.gym_env=="Ant-v4":
        env_xml = "ant.xml"
        args.frame_skip=5
        args.dt=0.01
    elif args.gym_env=="Hopper":
        env_xml = "hopper.xml"
        args.frame_skip=4
        args.dt=0.002
    elif args.gym_env=="Walker2d":
        env_xml = "walker2d.xml"
        args.frame_skip=4
        args.dt=0.002
    elif args.gym_env=="Humanoid-v4":
        env_xml = "humanoid.xml"
        args.frame_skip=5
        args.dt=0.003
    elif args.gym_env=="Swimmer":
        env_xml = "Swimmer.xml"
        args.frame_skip=4
        args.dt=0.01

    model_path=os.path.join(assets_dir,env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
    if args.gym_env=="Humanoid-v4":
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mjx_model = mjx.put_model(model)
    


mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = args.reward_fn_updates
DEMO_BATCH = args.N_steps
sample_trajs = []

D_demo, D_samp = np.array([]), jnp.array([])


#D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
#D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
return_list, sum_of_cost_list = [], []

mpc_method = "Single_FIM_3D_action_NN_MPPI"


seeds=np.array([args.seed])
args.runs=np.shape(seeds)[0]
epoch_cost_runs=[]
expert_cost_runs=[]
epoch_cost_dir = osp.join('results',args.gym_env)
dones = jnp.zeros((args.N_steps,), dtype=jnp.float32)
dones = dones.at[-1].set(1.0)  # mark last step as terminal

for runs in range(args.runs):
    args.seed = int(seeds[runs])
    print(args.seed)

    epoch_cost = []
    expert_cost = []

    for i in range(args.rirl_iterations):

        # ============================================================
        # INITIALIZATION (ONLY ON ITERATION 0)
        # ============================================================
        if i == 0:
            demo_generator = GenerateDemo(args.gym_env, max_frames=args.N_steps_expert)
            states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(args.seed)
            print("rewards_demo:", rewards_demo)

            args.DEMO_BATCH = min(DEMO_BATCH, states_d.shape[0])

            args.a_dim = actions_d.shape[-1]
            args.s_dim = states_d.shape[-1]

            # -----------------------------
            # Initialize reward function NN
            # -----------------------------
            cost_f = CostNN(state_dims=args.s_dim, hidden_dim=args.hidden_dim)
            init_rng = jax.random.PRNGKey(0)
            variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
            params = variables['params']
            tx = optax.adam(args.lr)
            state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
            def cost_function(state, state_train):
                return state_train.apply_fn(
                    {'params': state_train.params},
                    state.reshape(1, -1)
                ).ravel()

            # -------------------------------------------------------
            # Prepare expert demos D_demo = [state, prob, action]
            # -------------------------------------------------------
            D_demo = preprocess_traj([[states_d, actions_d, actions_d]], np.array([]), is_Demo=True)
            D_demo = jnp.concatenate((D_demo[:, :args.s_dim], D_demo[:, args.s_dim:]), axis=1)

            # =============================================
            #  PPO SETUP (if args.PPO == True)
            # =============================================
            if args.PPO:
                model_p = policy_model(action_dim=args.a_dim)
                dummy_input = jnp.zeros((1, args.s_dim))
                params_p = model_p.init(init_rng, dummy_input)['params']
                tx = optax.adam(3e-4)
                state_train_p = train_state.TrainState.create(apply_fn=model_p.apply, params=params_p, tx=tx)

                policy = PPOPolicy(
                    action_dim=args.a_dim,
                    mjx_model=mjx_model,
                    dynamics=get_step_model(args.gym_env, env),
                    policy_model=state_train_p,
                    policy_net=model_p,
                    args=args,
                )

            # ===================================================
            #  USE FAST MPPI (THIS IS THE IMPORTANT PART)
            # ===================================================
            else:
                u_min, u_max = get_action_space(args.gym_env, env)
                u_min = jnp.asarray(u_min, dtype=jnp.float32)
                u_max = jnp.asarray(u_max, dtype=jnp.float32)

                sigmas = get_action_cov(args.gym_env, env)

                fast_mppi = FastMPPI(
                    mjx_model=mjx_model,
                    dim_state=args.s_dim,
                    dim_control=args.a_dim,
                    u_min=u_min,
                    u_max=u_max,
                    sigmas=sigmas,
                    lambda_=args.lambda_,
                    horizon=args.horizon,
                    num_samples=args.num_traj,
                    frame_skip=args.frame_skip,
                    exploration=0.1,
                    seed=args.seed,
                    cost_function=cost_function,   # <-- FIXED
                )



            # --------------------------------------
            # RGCL initialization
            # --------------------------------------
            if args.rgcl:
                flat_params, treedef = jax.tree_util.tree_flatten(params)
                theta = jnp.concatenate([p.reshape(-1) for p in flat_params])
                n_theta = len(theta)
                P_theta = args.P * jnp.eye(n_theta)

        # ================================================================
        # RUN ONE EPISODE USING FAST MPPI (or PPO)
        # ================================================================

        initial_state = D_demo[0, :args.s_dim]

        if args.rgcl:
            # Your RGCL implementation stays unchanged
            start = time.time()
            traj = policy.RGCL_lax(args, params, state_train, initial_state, D_demo, P_theta, thetas)
            end = time.time()

            rewards = traj[-3]
            params = traj[-1]
            total_cost = rewards
            print(f"[RGCL] Time: {end - start:.4f}s  Reward {rewards:.4f}")

        elif args.online:
            # Online AIRL/GCL etc.
            traj = policy.generate_session(args, state_train, initial_state, thetas)
            rewards = traj[-2]
            state_train = traj[-1]
            total_cost = rewards

        else:
            # =====================================================
            # THIS IS THE MPPI VERSION (FAST)
            # =====================================================
            if args.PPO:
                traj = policy.generate_session(args, D_demo)
                rewards = traj[-2]
                total_cost = rewards
                D_samp = preprocess_traj([traj], np.array([]))

            else:
                # ---------------------------------------------
                # CALL FAST MPPI HERE INSTEAD OF OLD MPPI
                # ---------------------------------------------
                start = time.time()
                traj = generate_session_fast_mppi(
                    fast_mppi,
                    mjx_model,
                    state_train,
                    D_demo,
                    args,
                    reward_fn=get_cost,
                )
                end = time.time()

                rewards = traj[-1]
                total_cost = rewards

                print(f"[FastMPPI] Time: {end - start:.4f}s  Reward {rewards:.4f}")

                D_samp = preprocess_traj([traj], np.array([]))

        # ===================================================
        # UPDATE REWARD NETWORK (AIRL/GAIL/GCL/SQIL)
        # ===================================================
        if not args.rgcl and not args.online:
            loss_rew_list = []

            for _ in range(args.reward_fn_updates):
                # Select samples
                D_s_samp = D_samp
                D_s_demo = D_demo

                states, probs, actions = D_s_samp[:, :args.s_dim], D_s_samp[:, args.s_dim], D_s_samp[:, args.s_dim+1:]
                states_expert, probs_expert, actions_expert = (
                    D_s_demo[:, :args.s_dim],
                    D_s_demo[:, args.s_dim],
                    D_s_demo[:, args.s_dim + 1:],
                )

                # Run AIRL / GAIL / GCL / SQIL loss
                if args.airl:
                    grads, loss_IOC = apply_model_AIRL(state_train, states, actions, states_expert, actions_expert, probs, probs_expert, args.UB)
                elif args.sqil:
                    grads, loss_IOC = apply_model_SQIL(state_train, states, actions, states_expert, actions_expert, probs, probs_expert)
                else:
                    grads, loss_IOC = apply_model(state_train, states, actions, states_expert, actions_expert, probs, probs_expert, args.UB)

                state_train = update_model(state_train, grads)
                loss_rew_list.append(loss_IOC)

            mean_loss_rew.append(jnp.mean(jnp.array(loss_rew_list)))

        # ===================================================
        # LOGGING
        # ===================================================
        epoch_cost.append(total_cost)
        expert_cost.append(rewards_demo)

        if i % 10 == 0:
            save_dir = f"{epoch_cost_dir}/{method}"
            os.makedirs(save_dir, exist_ok=True)

            np.save(f"{save_dir}/cost_{10*i}_seed={args.seed}.npy", epoch_cost)
            np.save(f"{save_dir}/expert_cost_{10*i}_seed={args.seed}.npy", expert_cost)

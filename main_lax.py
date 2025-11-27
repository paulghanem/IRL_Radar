# %%
# Configure CUDA library paths for JAX - must be done BEFORE importing JAX
import os
import sys
import glob

# Set JAX environment variables
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Find and configure CUDA library paths from site-packages
site_packages = None
for path in sys.path:
    if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
        site_packages = path
        break

if site_packages:
    nvidia_path = os.path.join(site_packages, 'nvidia')
    subdirs = [d for d in os.listdir(nvidia_path) if os.path.isdir(os.path.join(nvidia_path, d))]

    lib_paths = []
    for subdir in subdirs:
        lib_path = os.path.join(nvidia_path, subdir, 'lib')
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)

    if lib_paths:
        existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(lib_paths)
        if existing_ld_path:
            new_ld_path = f"{new_ld_path}:{existing_ld_path}"
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

# Now import the rest
from flax.training import train_state,checkpoints

import flax
import optax
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
from src.control.mppi_class import MPPI
from src.control.PPO import PPOPolicy,policy_model
from src.control.dynamics import get_action_cov,get_action_space,get_step_model

from utils.helpers import GenerateDemo

import gymnax
#from brax import envs
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
parser.add_argument("--N_steps_expert",default=200,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--N_steps",default=200,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--rirl_iterations",default=9,type=int,help="The number of epoch updates")
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
parser.add_argument('--gym_env', default="CartPole-v1",type=str, help='gym environment to test (CartPole-v1 , Pendulum-v1)')
parser.add_argument('--PPO', action=argparse.BooleanOptionalAction,default=False,type=bool, help='PPO policy flag')

parser.add_argument("--online",action=argparse.BooleanOptionalAction,default=False,type=bool,help="online version of bechmarks ")

parser.add_argument("--diagonal",action=argparse.BooleanOptionalAction,default=False,type=bool,help="diagonal version of hessians ")

# ==================== MPPI CONFIGURATION ======================== #
parser.add_argument('--horizon', default=50,type=int, help='Horizon for MPPI control')
parser.add_argument('--num_traj', default=2000,type=int, help='Number of MPPI control sequences samples to generate')




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
        #env_brax = envs.get_environment('halfcheetah')
    elif args.gym_env=="Ant-v4":
        env_xml = "ant.xml"
        args.frame_skip=5
        args.dt=0.01
        #env_brax = envs.get_environment('Ant')
    elif args.gym_env=="Hopper":
        env_xml = "hopper.xml"
        args.frame_skip=4
        args.dt=0.002
        #env_brax = envs.get_environment('Hopper')
    elif args.gym_env=="Walker2d":
        env_xml = "walker2d.xml"
        args.frame_skip=4
        args.dt=0.002
       # env_brax = envs.get_environment('walker2d',exclude_current_positions_from_observation=False)
    elif args.gym_env=="Humanoid-v4":
        env_xml = "humanoid.xml"
        args.frame_skip=5
        args.dt=0.003
        #env_brax = envs.get_environment('Humanoid')
    elif args.gym_env=="Swimmer":
        env_xml = "Swimmer.xml"
        args.frame_skip=4
        args.dt=0.01
        #env_brax = envs.get_environment('Swimmer')

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
for runs in range (args.runs):
    args.seed = int(seeds[runs])
    print(args.seed)
    epoch_cost=[]
    expert_cost=[]
    
    for i in range(args.rirl_iterations):
        print("\n" + "=" * 70)
        print(f"RIRL ITERATION {i+1} / {args.rirl_iterations}")
        print("=" * 70)
        if i== 0:
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
           # fname = f"expert_demo_{args.gym_env}_seed{args.seed}.npz"
           # data = np.load(fname)
           # states_d = data["states"]
           # actions_d = data["actions"]
           # rewards_demo = data["rewards"]
            #env=data["env"]
            demo_generator = GenerateDemo(args.gym_env,max_frames=args.N_steps_expert)
            states_d,actions_d,rewards_demo,env = demo_generator.generate_demo(args.seed)
            print("rewards_demo",rewards_demo)
            # if args.gym_env=="Ant":
            #     states_d=states_d[:,:27]
            # if args.gym_env=="Humanoid-v4":
            #     states_d=states_d[:,:47]
           
    
            args.DEMO_BATCH = min(DEMO_BATCH,states_d.shape[0])
            #args.N_steps = min(DEMO_BATCH,states_d.shape[0])
    
            # initalize Neural Network...
            
            args.a_dim = actions_d.shape[-1]
            args.s_dim = states_d.shape[-1]
            thetas = jnp.ones((1, args.s_dim))
    
            # INITILIZING POLICY AND REWARD FUNCTION
            u_min, u_max = get_action_space(args.gym_env,env)
            cov_scaler = get_action_cov(args.gym_env,env)
    
    
    
    
            # policy = P_MPPI((args.s_dim,),  args.a_dim,args=args)
            cost_f = CostNN(state_dims=args.s_dim,hidden_dim=args.hidden_dim) #CostNN(state_dims=args.s_dim)
            @jax.jit
            def cost_function(state,state_train):
    
                return state_train.apply_fn({'params':state_train.params},state.reshape(1,-1)).ravel()
    
    
            u_min, u_max = get_action_space(args.gym_env,env)
            cov_scaler = get_action_cov(args.gym_env,env)
            
            init_rng = jax.random.key(0)
    
            
            
            # if args.PPO:
            #     model_p=policy_model(action_dim=args.a_dim)
            #     dummy_input = jnp.zeros((1, args.s_dim))  # (batch, obs)
                
            #     params_p = model_p.init(init_rng, dummy_input)['params']
            #     #variables_p = model_p.init(init_rng, jnp.ones((1, args.s_dim)))
        
            #     #params_p = variables_p['params']
            #     # params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
            #     # params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
            #     tx = optax.adam(learning_rate=3e-4)
            #     state_train_p = train_state.TrainState.create(apply_fn=model_p.apply, params=params_p, tx=tx)
            #     policy= PPOPolicy(action_dim=args.a_dim, mjx_model=mjx_model,dynamics=get_step_model(args.gym_env,env),policy_model=state_train_p,policy_net=model_p,args=args)

              
            
               
 
                    
            # cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
            
    
            variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
    
            params = variables['params']
            # params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
            # params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
            tx = optax.adam(learning_rate=args.lr)
            state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
            policy = MPPI(
                state_train=state_train,
                horizon=args.horizon,
                num_samples=args.num_traj,
                # subiterations=args.MPPI_iterations,
                dim_state=args.s_dim,
                dim_control=args.a_dim,
                dynamics=get_step_model(args.gym_env,env),
                cost_func=jax.jit(vmap(cost_function,in_axes=(0,None))),
                u_min=u_min,
                u_max=u_max,
                sigmas=cov_scaler,
                lambda_=args.lambda_,
                env=env,
                mjx_model=mjx_model,
                gym_env=args.gym_env,
                use_mujoco=True
            )
               
            D_demo=np.array([])
        
            demo_trajs=[[states_d,actions_d,actions_d]]
            D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
            D_demo=jnp.concatenate((D_demo[:,:args.s_dim],D_demo[:,args.s_dim:]),axis=1)
            
            if args.rgcl:
                flat_params, treedef = jax.tree_util.tree_flatten(params)
                theta=jnp.concatenate([p.flatten() for p in flat_params])
                #pdb.set_trace()
                n_theta = len(theta)
                P_theta = args.P * jnp.identity(n_theta)
        steps=0 
        initial_state=D_demo[0,:args.s_dim]
       
    
        if args.rgcl:
            start = time.time()
            #trajs = [policy.RGCL(args,params,state_train,initial_state,D_demo[steps:steps+args.N_steps,:],P_theta,thetas)]
            trajs = [policy.RGCL_lax(args,params,state_train,initial_state,D_demo,P_theta,thetas)]
            end = time.time()
            rewards=trajs[0][-3]
            #P_theta=trajs[0][-2]
            params=trajs[0][-1]
            #print(P_theta)
            total_cost=rewards
            #pdb.set_trace()
            print(f"Execution time: {end - start:.4f} seconds,Total True Reward = {rewards:.4f}")
            
        elif args.online:
            trajs = [policy.generate_session(args,state_train,initial_state,thetas)]
            rewards=trajs[0][-2]
            state_train=trajs[0][-1]
            total_cost=rewards
        else:
            
            if args.PPO:
                trajs = [policy.generate_session(args,D_demo)]
                rewards=trajs[0][-2]
                total_cost=rewards
                sample_trajs = [trajs[0][:-2]] #+ sample_trajs
                log_probs_old=trajs[0][-1]
                #sample_trajs = demo_trajs + sample_trajs
                D_samp=np.array([])
                D_samp = preprocess_traj(trajs, D_samp)
                
            else:
                start = time.time()
                #trajs = [policy.generate_session(args,state_train,initial_state,D_demo,thetas)]
                trajs=[policy.generate_session_lax(args,state_train,D_demo)]
                #trajs=[policy.generate_session_loop(args,state_train,D_demo)]
                end = time.time()





                rewards=trajs[0][-1]

                print(f"\nIteration {i+1} Trajectory Generation:")
                print(f"  Execution time (LAX scan): {end - start:.4f} seconds")
                print(f"  Total reward: {rewards:.4f}")

                # Analyze states and actions
                states_arr = jnp.array(trajs[0][0])
                actions_arr = jnp.array(trajs[0][2])

                print(f"  Total steps: {len(trajs[0][0])}")
                print(f"  Average reward per step: {rewards / len(trajs[0][0]):.4f}")

                # First and last states
                print(f"\n  First 3 states (first 6 dims):")
                for idx in range(min(3, len(trajs[0][0]))):
                    print(f"    Step {idx}: {states_arr[idx][:6]}")

                print(f"\n  Last 3 states (first 6 dims):")
                for idx in range(max(0, len(trajs[0][0])-3), len(trajs[0][0])):
                    print(f"    Step {idx}: {states_arr[idx][:6]}")

                # Check for NaN/Inf
                has_nan = jnp.any(jnp.isnan(states_arr))
                has_inf = jnp.any(jnp.isinf(states_arr))
                print(f"\n  Validity checks:")
                print(f"    NaN values: {'YES - WARNING!' if has_nan else 'No'}")
                print(f"    Inf values: {'YES - WARNING!' if has_inf else 'No'}")

                total_cost=rewards
                sample_trajs = [trajs[0][:-1]] #+ sample_trajs
                #sample_trajs = demo_trajs + sample_trajs
                D_samp=np.array([])
                D_samp = preprocess_traj(trajs, D_samp)
               # print(steps,f"rewards: {rewards:.4f} ")
        
        #D_samp = D_demo
        
        
        initial_state=D_demo[steps,:args.s_dim]
        # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
        if not args.rgcl and not args.online:
            loss_rew = []
            for _ in range(REWARD_FUNCTION_UPDATE):
                selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
                #selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)
                selected_demo=D_demo[steps-args.N_steps:steps]
    
                #D_s_samp = D_samp[selected_samp]
                #D_s_demo = D_demo[selected_demo]
                D_s_samp = D_samp
                D_s_demo = D_demo
                #D̂ samp ← D̂ demo ∪ D̂ samp
                #D_s_samp = jnp.concatenate((D_s_demo, D_s_samp), axis = 0)
    
                states, probs, actions = D_s_samp[:,:args.s_dim], D_s_samp[:,args.s_dim], D_s_samp[:,args.s_dim+1:]
                states_expert,probs_experts, actions_expert = D_s_demo[:,:args.s_dim], D_s_demo[:,args.s_dim], D_s_demo[:,args.s_dim+1:]
    
                # Reducing from float64 to float32 for making computaton faster
                #states = torch.tensor(states, dtype=torch.float32)
                #probs = torch.tensor(probs, dtype=torch.float32)
                #actions = torch.tensor(actions, dtype=torch.float32)
                #states_expert = torch.tensor(states_expert, dtype=torch.float32)
                #actions_expert = torch.tensor(actions_expert, dtype=torch.float32)
                if args.airl:
                    grads, loss_IOC = apply_model_AIRL(state_train, states, actions,states_expert,actions_expert,probs,probs_experts,args.UB)
                
                elif args.sqil:
                     grads, loss_IOC = apply_model_SQIL(state_train, states, actions,states_expert,actions_expert,probs,probs_experts)
                
                else :
                    grads, loss_IOC = apply_model(state_train, states, actions,states_expert,actions_expert,probs,probs_experts,args.UB)
    
                state_train = update_model(state_train, grads)
    
    
    
                loss_rew.append(loss_IOC)
                next_states = jnp.vstack([states[1:], states[-1:]]) 
                if args.PPO:
                    policy.update_ppo(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    log_probs_old=log_probs_old,
                    next_states=next_states,
                    batch_size=args.N_steps,
                    value_fn=None  # or your critic if available
                    )
            
            
            # mean_costs.append(np.mean(sum_of_cost_list))
            mean_loss_rew.append(np.mean(loss_rew))
           
                        
                
        
        epoch_cost.append(total_cost)
        expert_cost.append(rewards_demo)

        # Saving disabled for CartPole testing
        # if np.remainder(i,10)==0:
        #     save_dir = f"{epoch_cost_dir}/{method}"
        #     os.makedirs(save_dir, exist_ok=True)
        #     np.save(f"{save_dir}/cost_{10* i}_seed={args.seed}_lambda={args.lambda_}_horizon={args.horizon}_trajectories={args.num_traj}_Q={args.Q}_P={args.P}_ndim={args.hidden_dim}.npy",epoch_cost)
        #     np.save(f"{save_dir}/expert_cost_{10* i}_seed={args.seed}_lambda={args.lambda_}_horizon={args.horizon}_trajectories={args.num_traj}_Q={args.Q}_P={args.P}_ndim={args.hidden_dim}.npy",expert_cost)
            #np.save(osp.join(epoch_cost_dir,method+'_epoch_cost.npy'), epoch_cost_runs)
            #np.save(osp.join(epoch_cost_dir,method+'_expert_cost.npy'), expert_cost_runs)

   
    #epoch_cost_runs.append(epoch_cost)
    #expert_cost_runs.append(expert_cost)
 






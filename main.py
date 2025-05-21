

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
parser.add_argument("--N_steps",default=200,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--rirl_iterations",default=100,type=int,help="The number of epoch updates")
parser.add_argument("--reward_fn_updates",default=10,type=int,help="The number of reward fn updates")
parser.add_argument("--hidden_dim",default=16,type=int,help="The number of hidden neurons")
parser.add_argument("--lambda_",default=0.01,type=float,help="Temperature in MPPI (lower makers sharper)")
parser.add_argument("--runs",default=1,type=int,help="The number of runs")

parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')


parser.add_argument('--lr', default=1e-4,type=float, help='learning rate')
parser.add_argument('--P', default=1e-1,type=float, help='rgcl initial covariance')
parser.add_argument('--Q', default=1e-4,type=float, help='rgcl learning rate')

parser.add_argument("--UB",action=argparse.BooleanOptionalAction,default=False,type=bool,help="Upper bound loss  ")
parser.add_argument('--sqil', action=argparse.BooleanOptionalAction,default=False,type=bool, help='sqil method flag (automatically turns sqil flag on)')
parser.add_argument('--gail', action=argparse.BooleanOptionalAction,default=False,type=bool, help='gail method flag (automatically turns gail flag on)')
# %%
parser.add_argument('--airl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='airl method flag')

parser.add_argument('--rgcl', action=argparse.BooleanOptionalAction,default=False,type=bool, help='rgcl method flag')
parser.add_argument('--gym_env', default="CartPole-v1",type=str, help='gym environment to test (CartPole-v1 , Pendulum-v1)')

parser.add_argument("--online",action=argparse.BooleanOptionalAction,default=False,type=bool,help="online version of bechmarks ")


# ==================== MPPI CONFIGURATION ======================== #
parser.add_argument('--horizon', default=10,type=int, help='Horizon for MPPI control')
parser.add_argument('--num_traj', default=25,type=int, help='Number of MPPI control sequences samples to generate')




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
if args.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d"]:
    if args.gym_env=="HalfCheetah-v4":
        env_xml = "half_cheetah.xml"
    elif args.gym_env=="Ant":
        env_xml = "ant.xml"
    elif args.gym_env=="Hopper":
        env_xml = "hopper.xml"
    elif args.gym_env=="Walker2d":
        env_xml = "walker2d.xml"

    model_path=os.path.join(assets_dir,env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
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
for runs in range (args.runs):
    args.seed = int(seeds[runs])
    print(args.seed)
    epoch_cost=[]
    expert_cost=[]
    
    for i in range(args.rirl_iterations):
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
            demo_generator = GenerateDemo(args.gym_env,max_frames=args.N_steps)
            states_d,actions_d,rewards_demo,env = demo_generator.generate_demo(args.seed)
            if args.gym_env=="Ant":
                states_d=states_d[:,:27]
           
    
            args.DEMO_BATCH = min(DEMO_BATCH,states_d.shape[0])
            args.N_steps = min(DEMO_BATCH,states_d.shape[0])
    
            # initalize Neural Network...
            
            args.a_dim = actions_d.shape[-1]
            args.s_dim = states_d.shape[-1]
            thetas = jnp.ones((1, args.s_dim))
    
            # INITILIZING POLICY AND REWARD FUNCTION
            u_min, u_max = get_action_space(args.gym_env,env)
            cov_scaler = get_action_cov(args.gym_env,env)
    
    
    
    
            # policy = P_MPPI((args.s_dim,),  args.a_dim,args=args)
            cost_f = CostNN(state_dims=args.s_dim,hidden_dim=args.hidden_dim) #CostNN(state_dims=args.s_dim)
    
            def cost_function(state,state_train):
    
                return state_train.apply_fn({'params':state_train.params},state.reshape(1,-1)).ravel()
    
    
            u_min, u_max = get_action_space(args.gym_env,env)
            cov_scaler = get_action_cov(args.gym_env,env)
            
    
            policy = MPPI(
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
                gym_env=args.gym_env
            )
    
            # cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
            init_rng = jax.random.key(0)
    
            variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
    
            params = variables['params']
            # params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
            # params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
            tx = optax.adam(learning_rate=args.lr)
            state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
    
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
            
       
        if args.rgcl:
            
            trajs = [policy.RGCL(args,params,state_train,D_demo,P_theta,thetas)]
            rewards=trajs[0][-2]
            P_theta=trajs[0][-1]
            print(P_theta)
            total_cost=rewards
        elif args.online:
            trajs = [policy.generate_session(args,state_train,D_demo,thetas)]
            rewards=trajs[0][-2]
            state_train=trajs[0][-1]
            total_cost=rewards
        else:
            trajs = [policy.generate_session(args,state_train,D_demo,thetas)]
            rewards=trajs[0][-1]
            total_cost=rewards
            sample_trajs = [trajs[0][:-1]] #+ sample_trajs
            #sample_trajs = demo_trajs + sample_trajs
            D_samp=np.array([])
            D_samp = preprocess_traj(trajs, D_samp)
        
        #D_samp = D_demo
    
        # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
        if not args.rgcl and not args.online:
            loss_rew = []
            for _ in range(REWARD_FUNCTION_UPDATE):
                selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
                selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)
    
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
            
            
            # mean_costs.append(np.mean(sum_of_cost_list))
            mean_loss_rew.append(np.mean(loss_rew))
       
        
        epoch_cost.append(total_cost)
        expert_cost.append(rewards_demo)
       
        if np.remainder(i,10)==0:
            save_dir = f"{epoch_cost_dir}/{method}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/cost_{10* i}_seed={args.seed}_lambda={args.lambda_}_horizon={args.horizon}_trajectories={args.num_traj}_Q={args.Q}_P={args.P}_ndim={args.hidden_dim}.npy",epoch_cost)
            np.save(f"{save_dir}/expert_cost_{10* i}_seed={args.seed}_lambda={args.lambda_}_horizon={args.horizon}_trajectories={args.num_traj}_Q={args.Q}_P={args.P}_ndim={args.hidden_dim}.npy",expert_cost)
            #np.save(osp.join(epoch_cost_dir,method+'_epoch_cost.npy'), epoch_cost_runs)
            #np.save(osp.join(epoch_cost_dir,method+'_expert_cost.npy'), expert_cost_runs)

   
    #epoch_cost_runs.append(epoch_cost)
    #expert_cost_runs.append(expert_cost)
 



#os.makedirs(epoch_cost_dir,exist_ok=True)
# Save the array
#np.save(osp.join(epoch_cost_dir,method+'_epoch_cost.npy'), epoch_cost_runs)
#np.save(osp.join(epoch_cost_dir,method+'_expert_cost.npy'), expert_cost_runs)

# just for cartpole...

# visualization_irl = [policy.generate_session(args,state_train,D_demo,mpc_method,thetas)]
# states_mppi_irl = [get_state(state=state,action=action,time=i,env_name=args.gym_env) for i,(state,action) in enumerate(zip(visualization_irl[0][0],visualization_irl[0][1]))]
# costs_mppi_irl = [get_cost(args.gym_env)(state) for state in visualization_irl[0][0]]

# from utils.helpers import load_config
# policy_method_agent = "es" if args.gym_env == "MountainCarContinuous-v0" else "ppo"
# base = osp.join("expert_agents",args.gym_env, policy_method_agent)
# configs = load_config(base + ".yaml")

# env, env_params = gymnax.make(
#     configs.train_config.env_name,
#     **configs.train_config.env_kwargs,
# )

# vis = Visualizer(env, env_params, states_mppi_irl, np.array(costs_mppi_irl))
# vis.animate(osp.join("results",f"{args.gym_env}-mppi-irl.gif"))

# print("Simulation completed. Check the 'results' folder for the recorded episodes.")


# vis = Visualizer(env, env_params, states_mppi_irl, np.array(costs_mppi_irl))
# vis.animate(osp.join("results",f"{args.gym_env}-mppi-irl.gif"))


# config = {'dimensions': np.array([5, 3])}
# ckpt_single = {'model_single': state_train, 'config': config, 'data': [D_samp]}
# checkpoints.save_checkpoint(ckpt_dir='/tmp/flax_ckpt/flax-checkpointing',
#                             target=ckpt_single,
#                             step=0,
#                             overwrite=True,
#                             keep=2)



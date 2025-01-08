# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:38:21 2024

@author: siliconsynapse
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:00:04 2024

@author: siliconsynapse
"""



from flax.training import train_state,checkpoints
import flax 
import optax

import os
import argparse
import os.path as osp

import numpy as np
import jax.numpy as jnp
import jax


from experts.P_MPPI import P_MPPI
from cost_jax import CostNN, apply_model, update_model

from src.objective_fns.objectives import *
from utils.helpers import generate_demo,load_config
from utils.models import load_neural_network

import gymnax
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
parser.add_argument("--N_steps",default=150,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
parser.add_argument('--remove_tmp_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to remove tmp images? --remove_tmp_images for yes --no-remove_tmp_images for no')
parser.add_argument('--tail_length',default=10,type=int,help="The length of the tail of the radar trajectories in plottings")
parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')
parser.add_argument('--fim_method_policy', default="SFIM_NN",type=str, help='FIM Calculation [SFIM,PFIM]')
parser.add_argument('--fim_method_demo', default="SFIM",type=str, help='FIM Calculation [SFIM,PFIM]')
parser.add_argument('--gail', default=False,type=bool, help='gail metod flag')


# ==================== MPPI CONFIGURATION ======================== #
parser.add_argument('--horizon', default=50,type=int, help='Horizon for MPPI control')
parser.add_argument('--num_traj', default=250,type=int, help='Number of MPPI control sequences samples to generate')
parser.add_argument('--MPPI_iterations', default=75,type=int, help='Number of MPPI sub iterations (proposal adaptations)')





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


n_actions = 1

state_shape =((4,))


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
DEMO_BATCH = 150
sample_trajs = []

D_demo, D_samp = np.array([]), jnp.array([])


#D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
#D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
return_list, sum_of_cost_list = [], []

#U,chis,radar_states,target_states
thetas=jnp.ones((1,4))
mpc_method = "Single_FIM_3D_action_NN_MPPI"

for i in range(100):
    if (i== 0):
        base = osp.join("expert_agents", "CartPole-v1", "ppo")
        configs = load_config(base + ".yaml")
        model, model_params = load_neural_network(
            configs.train_config, base + ".pkl"
        )

        env, env_params = gymnax.make(
            configs.train_config.env_name,
            **configs.train_config.env_kwargs,
        )
        env_params.replace(**configs.train_config.env_params)


        states_d,actions_d,_ =generate_demo(
            env, env_params, model, model_params,max_frames=DEMO_BATCH,seed=args.seed)

        args.a_dim = actions_d.shape[-1]
        args.s_dim = states_d.shape[-1]

        D_demo=np.array([])
    
        demo_trajs=[[states_d,actions_d,actions_d]]
        D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
        D_demo=jnp.concatenate((D_demo[:,:state_shape[0]],D_demo[:,state_shape[0]:]),axis=1)



    trajs = [policy.generate_session(args,i,state_train,D_demo,mpc_method,thetas)]

   
    sample_trajs = trajs #+ sample_trajs
    #sample_trajs = demo_trajs + sample_trajs
    D_samp=np.array([])
    D_samp = preprocess_traj(trajs, D_samp)
    
    #D_samp = D_demo

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
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
        grads, loss_IOC = apply_model(state_train, states, actions,states_expert,actions_expert,probs,probs_experts)
        state_train = update_model(state_train, grads)
        


        loss_rew.append(loss_IOC)
    
    # mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))
    
    

config = {'dimensions': np.array([5, 3])}
ckpt_single = {'model_single': state_train, 'config': config, 'data': [D_samp]}
checkpoints.save_checkpoint(ckpt_dir='/tmp/flax_ckpt/flax-checkpointing',
                            target=ckpt_single,
                            step=0,
                            overwrite=True,
                            keep=2)
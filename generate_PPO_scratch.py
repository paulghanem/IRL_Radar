# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:18:48 2025

@author: siliconsynapse
"""
import argparse
import os
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import mujoco
from mujoco import mjx 
import gymnasium as gym
from src.control.PPO import PPOPolicy,policy_model,critic_model
from src.control.dynamics import get_action_cov,get_action_space,get_step_model
from utils.helpers import CustomTerminationWrapper



parser = argparse.ArgumentParser(description = 'PPO implementation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
parser.add_argument('--gym_env', default="HalfCheetah-v4",type=str, help='gym environment to test (CartPole-v1 , Pendulum-v1)')
parser.add_argument("--epochs",default=400,type=int,help="The number of epoch updates")
parser.add_argument("--N_steps",default=500000,type=int,help="The number of steps in the experiment in GYM ENV")
parser.add_argument("--evaluation_interval",default=10000,type=int,help="evaluation interval")
args = parser.parse_args()

mjx_model=None
assets_dir="assets"
if args.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]:
    if args.gym_env=="HalfCheetah-v4":
        env_xml = "half_cheetah.xml"
    elif args.gym_env=="Ant":
        env_xml = "ant.xml"
    elif args.gym_env=="Hopper":
        env_xml = "hopper.xml"
    elif args.gym_env=="Walker2d":
        env_xml = "walker2d.xml"
    elif args.gym_env=="Humanoid-v4":
        env_xml = "humanoid.xml"

    model_path=os.path.join(assets_dir,env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(model)


max_frames=2048




if args.gym_env in ["MountainCarContinuous-v0","CartPole-v1"]:
    env = CustomTerminationWrapper(gym.make(args.gym_env, render_mode='rgb_array'),max_steps=max_frames)
   

else:
    if args.gym_env  =="Ant":
        env = CustomTerminationWrapper(gym.make(args.gym_env ,exclude_current_positions_from_observation=True, render_mode='Human'),max_steps=max_frames)
    else:  
        env = CustomTerminationWrapper(gym.make(args.gym_env ,exclude_current_positions_from_observation=False, render_mode='Human'),max_steps=max_frames)

obs, info = env.reset(seed=args.seed)

if isinstance(env.action_space, gym.spaces.Discrete):
    args.a_dim = env.action_space.n
else:
    args.a_dim = env.action_space.shape[0]
args.s_dim= env.observation_space.shape[0] 

model_p=policy_model(action_dim=args.a_dim)
dummy_input = jnp.zeros((1, args.s_dim))  # (batch, obs)
init_rng = jax.random.key(0)

dones = jnp.zeros((args.evaluation_interval,), dtype=jnp.float32)
dones = dones.at[-1].set(1.0)  # mark last step as terminal

params_p = model_p.init(init_rng, dummy_input)['params']
#variables_p = model_p.init(init_rng, jnp.ones((1, args.s_dim)))

#params_p = variables_p['params']
# params['Dense_0']['bias']=jnp.ones(params['Dense_0']['bias'].shape)
# params['Dense_0']['kernel']=jnp.identity(params['Dense_0']['kernel'].shape[0])
tx = optax.adam(learning_rate=3e-4)
state_train_p = train_state.TrainState.create(apply_fn=model_p.apply, params=params_p, tx=tx)

model_c=critic_model()
dummy_input = jnp.zeros((1, args.s_dim))  # (batch, obs)
params_c = model_c.init(init_rng, dummy_input)['params']
tx_c = optax.adam(learning_rate=3e-4)
state_train_c = train_state.TrainState.create(apply_fn=model_c.apply, params=params_c, tx=tx_c)
policy= PPOPolicy(action_dim=args.a_dim, mjx_model=mjx_model,dynamics=get_step_model(args.gym_env,env),policy_model=state_train_p,policy_net=model_p,value_fn=state_train_c,args=args)

step=0
while step < args.N_steps:

    x0=obs.reshape((1,-1))
    
    trajs = [policy.generate_session_lax(args,x0)]
    states, traj_probs, actions,rewards,log_probs_old=trajs[0]
    states=jnp.array(states)
    traj_probs=jnp.array(traj_probs)
    actions=jnp.array(actions)
    rewards=jnp.array(rewards)
    log_probs_old=jnp.array(log_probs_old)
    next_states = jnp.vstack([states[1:], states[-1:]]) 
    
    policy.update_ppo(states=states, actions=actions, rewards=rewards,
    dones=dones, log_probs_old=log_probs_old, next_states=next_states,num_epochs=10)  # or your critic if available
    
    step+=args.evaluation_interval
    obs=states[-1]
    print(step,jnp.sum(rewards))
    


# for _ in range(TIMESTEPS):
#     action, _states = model.predict(obs)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     env.render()
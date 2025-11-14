# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:24:13 2025

@author: siliconsynapse
"""

from src.control.dynamics import kinematics,kinematics_mujoco,kinematics_mujoco_lax
from src.control.dynamics import get_step_model
from utils.helpers import CustomTerminationWrapper
import os 
import mujoco
from mujoco import mjx 
import gymnasium as gym
import jax.numpy as jnp
import jax
import math



gym_env="HalfCheetah-v4"
max_frames=1

mjx_model=None
assets_dir="assets"
if gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]:
    if gym_env=="HalfCheetah-v4":
        env_xml = "half_cheetah.xml"
    elif gym_env=="Ant":
        env_xml = "ant.xml"
    elif gym_env=="Hopper":
        env_xml = "hopper.xml"
    elif gym_env=="Walker2d":
        env_xml = "walker2d.xml"
    elif gym_env=="Humanoid-v4":
        env_xml = "humanoid.xml"

    model_path=os.path.join(assets_dir,env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(model)
    
    
    mjx_data = mjx.make_data(mjx_model)
if gym_env in ["MountainCarContinuous-v0","CartPole-v1"]:
    env = CustomTerminationWrapper(gym.make(gym_env, render_mode='rgb_array'),max_steps=max_frames)
   

else:
    if gym_env  =="Ant":
        env = CustomTerminationWrapper(gym.make(gym_env ,exclude_current_positions_from_observation=True, render_mode='Human'),max_steps=max_frames)
    else:  
        env = CustomTerminationWrapper(gym.make(gym_env ,exclude_current_positions_from_observation=False, render_mode='Human'),max_steps=max_frames)

_dynamics=get_step_model(gym_env,env)

def step_mjx(state,action,mjx_model,mjx_data,_dynamics,gym_env,frame_skip):
    next_state=kinematics_mujoco(mjx_model,mjx_data,state.flatten(),action.reshape((1,-1)),_dynamics,gym_env,frame_skip=frame_skip).flatten()
    return next_state


#state=jnp.ones((18,1))
frame_skip=5

for i in [0.1,0.2,0.3,0.4,0.5,0.6,.7,.8]:
    action=i*jnp.ones((6,1))
    obs, info = env.reset(seed=42)
    state=obs
    
    next_state=step_mjx(state,action,mjx_model,mjx_data,_dynamics,gym_env,frame_skip=frame_skip)
    forward_reward=(next_state[0]-state[0])/(0.01*frame_skip)
    ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
    r = forward_reward - ctrl_cost
    obs, reward, terminated, truncated, info = env.step(action.flatten())
    #print("openai_gym: ",obs,"mjx: ",state)
    print("difference: ",obs-next_state)
    print("openai_gym_reward: ",reward,"mjx_reward: ",r)
    

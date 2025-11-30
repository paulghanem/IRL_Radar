# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:24:13 2025

@author: siliconsynapse
"""

from src.control.dynamics import kinematics,kinematics_mujoco
from src.control.dynamics import get_step_model
from utils.helpers import CustomTerminationWrapper
import os 
import mujoco
from mujoco import mjx 
import gymnasium as gym
import jax.numpy as jnp
import jax
import math
import pdb
import jax
# MUST be enabled for unit testing against CPU




gym_env="Swimmer"
max_frames=1

mjx_model=None
assets_dir="assets"
if gym_env in ["HalfCheetah-v4","Ant-v4","Hopper","Walker2d","Humanoid-v4","Swimmer"]:
    if gym_env=="HalfCheetah-v4":
        env_xml = "half_cheetah.xml"
        a_dim=6
        frame_skip=5
        dt=0.01
    elif gym_env=="Ant-v4":
        env_xml = "ant.xml"
        a_dim=8
        frame_skip=5
        dt=0.01
    elif gym_env=="Hopper":
        env_xml = "hopper.xml"
        a_dim=3
        frame_skip=4
        dt=0.002
    elif gym_env=="Walker2d":
        env_xml = "walker2d.xml"
        a_dim=6
        frame_skip=4
        dt=0.002
    elif gym_env=="Humanoid-v4":
        env_xml = "humanoid.xml"
        a_dim=17
        frame_skip=5
        dt=0.003
    elif gym_env=="Swimmer":
        env_xml = "Swimmer.xml"
        a_dim=2
        frame_skip=4
        dt=0.01

    model_path=os.path.join(assets_dir,env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 1
    model.opt.ls_iterations = 1
    # model.opt.timestep = dt*frame_skip
    
    # dt=dt*frame_skip
    # frame_skip=1
    
    if gym_env=="Humanoid-v4":
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mjx_model = mjx.put_model(model)
    
    
    mjx_data = mjx.make_data(mjx_model)
if gym_env in ["MountainCarContinuous-v0","CartPole-v1"]:
    env = CustomTerminationWrapper(gym.make(gym_env, render_mode='rgb_array'),max_steps=max_frames)
   

else:
    if gym_env  =="Ant-v4":
        env = CustomTerminationWrapper(gym.make(gym_env ,exclude_current_positions_from_observation=False, render_mode='Human',use_contact_forces=False),max_steps=max_frames)
    else:  
        env = CustomTerminationWrapper(gym.make(gym_env ,exclude_current_positions_from_observation=False, render_mode='Human'),max_steps=max_frames)

_dynamics=get_step_model(gym_env,env)

def step_mjx(state,action,mjx_model,mjx_data,_dynamics,gym_env,frame_skip):
    next_state=kinematics_mujoco(mjx_model,mjx_data,state.flatten(),action.reshape((1,-1)),gym_env,frame_skip=frame_skip).flatten()
    return next_state

def reward_fn(gym_env, state, action,next_state, mjx_data,dt,frame_skip):
    #pdb.set_trace()
    forward_reward=(next_state[0]-state[0])/(dt)
   
    if gym_env == "CartPole-v1":
        x=next_state[0]
        x_threshold=2.4
        theta_cart=next_state[2]
        theta_threshold_radians=12 * 2 * math.pi / 360
        out_of_bounds = (x < -x_threshold) | (x > x_threshold)
        bad_angle = (theta_cart < -theta_threshold_radians) | (theta_cart > theta_threshold_radians)

        terminated = out_of_bounds | bad_angle
        #terminated = bool(
        # x < -x_threshold
        # or x > x_threshold
        # or theta_cart < -theta_threshold_radians
        # or theta_cart > theta_threshold_radians
        # )

        r = jnp.where(terminated, 0.0, 1.0)
    if gym_env == "Pendulum-v1":
        x=next_state[0]
        y=next_state[1]
        theta_pend=jnp.atan2(y,x)
        theta_dot=next_state[2]
        r = -(jnp.pow(theta_pend,2) + 0.1 * jnp.pow(theta_dot,2) + 0.001 * jnp.pow(action,2))
       
    if gym_env == "MountainCarContinuous-v0":
        r=-0.1 * jnp.pow(action,2)
        goal_position = 0.45
        goal_velocity = 0.0
        x=next_state[0]
        xd=next_state[1]
        r = r + jnp.where(x >= goal_position, 100.0, 0.0)
        # if goal_position <= x :
        #     r+=100 
    if gym_env == "HalfCheetah-v4":
        #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
        ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
        r = forward_reward - ctrl_cost
        #r=r.reshape((1,1))
        
    if gym_env == "Ant-v4":
        #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
        alive_bonus=1
        if next_state[2] <0.2 or next_state[2]>1:
            alive_bonus=0
        ctrl_cost = 0.5 * jnp.sum(jnp.square(action))
        r = forward_reward - ctrl_cost+alive_bonus
        
        
    if gym_env == "Hopper":
        #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
        alive_bonus=1
        if any(x < -100 for x in next_state[2:]) or any(x > 100 for x in next_state[2:]):
            alive_bonus=0
           # break
        if next_state[2] < -0.2  or next_state[2] > 0.2:
            alive_bonus=0
           # break 
        if next_state[1] < 0.7:
            alive_bonus=0
            #break  
       
        ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
        r = forward_reward - ctrl_cost + alive_bonus
        
        
        
    if gym_env == "Walker2d":
        alive_bonus = 1.0
    
        # JAX boolean: True when robot falls
        fall_cond = (
            (jnp.abs(next_state[2]) > 1.0) |
            (next_state[1] < 0.8) |
            (next_state[1] > 2.0)
        )
    
        # If fall_cond is True → alive_bonus = 0
        alive_bonus = jnp.where(fall_cond, 0.0, alive_bonus)
    
        ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
    
        r = forward_reward - ctrl_cost + alive_bonus
    
        
    if gym_env == "Humanoid-v4":
        #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
        alive_bonus=5
        if next_state[2] <1 or next_state[2]>2:
            alive_bonus=0
        #pdb.set_trace()
        quad_impact_cost = 0.5e-6 * jnp.square(mjx_data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
        r = 1.25*forward_reward - ctrl_cost  + alive_bonus
           
    if gym_env == "Swimmer":   
        ctrl_cost = 1e-4 * jnp.sum(jnp.square(action))
        r = forward_reward - ctrl_cost
        
        
    return r
#state=jnp.ones((18,1))


for i in [ -0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-.7,-.8,0.1,0.2,0.3,0.4,0.5,0.6,.7,.8]:
    action=i*jnp.ones((a_dim,1))
    obs, info = env.reset(seed=42)
    #obs=obs[:mjx_data.qpos.shape[0]+mjx_data.qvel.shape[0]]
    state=obs
    #pdb.set_trace()
    
    next_state=step_mjx(state,action,mjx_model,mjx_data,_dynamics,gym_env,frame_skip=frame_skip)
    #forward_reward=(next_state[0]-state[0])/(dt*frame_skip)
    r= reward_fn(gym_env, state, action, next_state, mjx_data,dt,frame_skip)


    obs, reward, terminated, truncated, info = env.step(action.flatten())
    obs=obs[:mjx_data.qpos.shape[0]+mjx_data.qvel.shape[0]]
    #print("openai_gym: ",obs,"mjx: ",state)
    print("difference: ",obs-next_state)
    print("openai_gym_reward: ",reward,"mjx_reward: ",r)
    

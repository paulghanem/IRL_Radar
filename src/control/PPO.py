# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 16:46:42 2025

@author: siliconsynapse
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from flax.training import train_state

import gymnax
import gymnasium as gym
from mujoco import mjx 
from tqdm.auto import tqdm
from src.control.dynamics import kinematics,kinematics_mujoco
from src.control.buffer import  RolloutBuffer
import math
import pdb

def mass_center(model,state):
    # Reshape body masses to (nbody, 1) for broadcasting
    mass = np.expand_dims(model.body_mass, axis=1)  # shape: (nbody, 1)
    
    # data.xpos contains the global positions of the body frames (shape: nbody x 3)
    xpos = state[:24]  # shape: (nbody, 3)

    # Compute center of mass as the weighted average of body positions
    com = np.sum(mass * xpos, axis=0) / np.sum(mass)
    
    # Return x-coordinate of the center of mass (you can return full com if needed)
    return com[0]

class policy_model(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim)(x)   # both μ and logσ
        log_std = self.param("log_std", nn.initializers.zeros, (1, self.action_dim))


        return mu, log_std
    
class critic_model(nn.Module):
   

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        mu = nn.Dense(1)(x)
        
        return mu

class PPOPolicy():
    def __init__(self,state_dim,action_dim,mjx_model, dynamics,policy_model,policy_net,args,rollout_length=2048, mix_buffer=20,value_fn=None):
        self.action_dim= action_dim
        self.state_dim=state_dim
        self.rollout_length=rollout_length
        self.args = args
        self.gym_env = args.gym_env
        self._dynamics = dynamics # set if needed
        self.mjx_model = mjx_model
        if self.mjx_model is not None :
            self.mjx_data = mjx.make_data(self.mjx_model)
        self.policy_model=policy_model
        self.policy_net=policy_net
        self.value_fn=value_fn
        
        self.max_steps=1000
        
        # Rollout buffer.
        self.buffer = RolloutBuffer.create(
            buffer_size=rollout_length,
            state_shape=(self.state_dim,),      # add parentheses
            action_shape=(self.action_dim,),    # add parentheses
            mix=mix_buffer
        )
        
        
        
    def reset(self):
        """
        Reset the previous action sequence.
        """
        self._previous_action_seq = jnp.zeros(
            (
                1, self.action_dim
            )
        )
        
    def reset_mjx_state(self, mjx_model, key=None, noise_scale=0.01):
        """
        Recreate a fresh mjx.Data object from a given mjx.Model.
        Optionally add small random noise to qpos/qvel for exploration.
    
        Args:
            mjx_model: Compiled MJX model.
            key: Optional PRNGKey for randomized initialization.
            noise_scale: Stddev of Gaussian noise added to qpos/qvel.
    
        Returns:
            mjx.Data: Freshly reset simulation state.
        """
        # Create a new mjx.Data object
        data = mjx.make_data(mjx_model)
    
        # Base zero initialization
        qpos0 = jnp.zeros_like(data.qpos)
        qvel0 = jnp.zeros_like(data.qvel)
    
        # If a random key is provided, add Gaussian noise to qpos/qvel
        if key is not None:
            key_qpos, key_qvel = jax.random.split(key)
            qpos0 = qpos0 + noise_scale * jax.random.normal(key_qpos, shape=qpos0.shape)
            qvel0 = qvel0 + noise_scale * jax.random.normal(key_qvel, shape=qvel0.shape)
    
        # Replace state fields
        data = data.replace(qpos=qpos0, qvel=qvel0)
    
        # Recompute derived quantities
        mjx.forward(mjx_model, data)
    
        return data


      
    def calculate_log_pi(self,log_stds, noises, actions):
        gaussian_log_probs = jnp.sum(-0.5 * jnp.power(noises, 2) - log_stds, axis=-1) - 0.5 * math.log(2 * math.pi) * log_stds.shape[-1]

        return gaussian_log_probs - jnp.sum( jnp.log(1 - jnp.power(actions, 2) + 1e-6), axis=-1)

    def atanh(self,x):
       return 0.5 * (jnp.log(1 + x + 1e-6) - jnp.log(1 - x + 1e-6))
   
    def evaluate_lop_pi(self,means, log_stds, actions):
        noises = (self.atanh(actions) - means) / (jnp.exp(log_stds) + 1e-8)
        return self.calculate_log_pi(log_stds, noises, actions)
    
    def evaluate_log_pi(self, states, actions,params):
        mu, log_std = self.policy_net.apply({'params': params}, states)
     
        return self.evaluate_lop_pi(mu, log_std, actions)
    
    def reward_fn(self,gym_env, state, action,forward_reward, mjx_data):
        
        if gym_env == "CartPole-v1":
            x=state[0]
            x_threshold=2.4
            theta_cart=state[2]
            theta_threshold_radians=12 * 2 * math.pi / 360
            terminated = bool(
            x < -x_threshold
            or x > x_threshold
            or theta_cart < -theta_threshold_radians
            or theta_cart > theta_threshold_radians
            )

            r= jnp.array([0.0])
            if not terminated:
                r= jnp.array([1.0])
        if gym_env == "Pendulum-v1":
            x=state[0]
            y=state[1]
            theta_pend=jnp.atan2(y,x)
            theta_dot=state[2]
            r = -(jnp.pow(theta_pend,2) + 0.1 * jnp.pow(theta_dot,2) + 0.001 * jnp.pow(action,2))
           
        if gym_env == "MountainCarContinuous-v0":
            r=-0.1 * jnp.pow(action,2)
            goal_position = 0.45
            goal_velocity = 0.0
            x=state[0]
            xd=state[1]
            if goal_position <= x :
                r+=100 
        if gym_env == "HalfCheetah-v4":
            #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
            ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost
            r=r.reshape((1,1))
            
        if gym_env == "Ant":
            #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
            alive_bonus=1
            if state[1] <0.2 or state[1]>1:
                alive_bonus=0
            ctrl_cost = 0.5 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost+alive_bonus
            r=r.reshape((1,1))
            
        if gym_env == "Hopper":
            #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
            alive_bonus=1
            if any(x < -100 for x in state[2:]) or any(x > 100 for x in state[2:]):
                alive_bonus=0
               # break
            if state[2] < -0.2  or state[2] > 0.2:
                alive_bonus=0
               # break 
            if state[1] < 0.7:
                alive_bonus=0
                #break  
           
            ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost + alive_bonus
            r=r.reshape((1,1))
            
        if gym_env == "Walker2d":
            #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
            alive_bonus=1
            if np.abs(state[2])>1 or state[1] <0.8 or state[1]>2:
                alive_bonus=0
            
            ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost + alive_bonus
            r=r.reshape((1,1))
            
        if gym_env == "Humanoid-v4":
            #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
            alive_bonus=5
            if state[2] <1 or state[2]>2:
                alive_bonus=0
            #pdb.set_trace()
            quad_impact_cost = 0.5e-6 * jnp.square(self.mjx_data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
            r = 1.25*forward_reward - ctrl_cost -quad_impact_cost + alive_bonus
            r=r.reshape((1,1))     
            
        
        return r
            
        

    def generate_session_lax(self, args, D_demo, frame_skip=1,dt=0.01,mpc_method=None, thetas=None):
        #self.mjx_data = self.reset_mjx_state(self.mjx_model)
   
        key = jax.random.PRNGKey(args.seed)
    
        # Initial state
        init_state = D_demo[0, :args.s_dim]
        reset_data = self.reset_mjx_state(self.mjx_model,key=key)
       
    
        def rollout_step(carry, t):
            state, key,buffer,mjx_data = carry
    
            # RNG split
            key, subkey = jax.random.split(key)
    
            # ---- PPO Policy Action Sampling ----
            mu, log_std = self.policy_net.apply({'params': self.policy_model.params}, state[None, :])
            noise = jax.random.normal(subkey, mu.shape)
            u = mu + jnp.exp(log_std) * noise
            action=jnp.tanh(u)
            logp=self.calculate_log_pi(log_std, noise, action)
            
            #state=kinematics_mujoco(self.mjx_model,self.mjx_data,state,action,self._dynamics,self.gym_env)
            state=jnp.array(state)
            action=jnp.array(action)
            next_state=kinematics_mujoco(self.mjx_model,self.mjx_data,state.flatten(),action.reshape((1,-1)),self._dynamics,self.gym_env,frame_skip=frame_skip).flatten()
          
           
            # ---- Dynamics update ----
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]: 
                if self.gym_env in ["Ant"]:
                    state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),next_state))
                
                # if self.gym_env=="Humanoid-v4":
                #     pos_before = mass_center(self.mjx_model,state)
                # if self.gym_env=="Humanoid-v4":
                #     pos_after = mass_center(self.mjx_model,state)
                # if self.gym_env in ["Ant"]:
                #     state=state[2:]
                #state=self._dynamics(self.mjx_model,self.mjx_data,state.flatten(), action_seq[0,:].flatten())
                if self.gym_env =="HalfCheetah-v4":
                    #self.mjx_data = self.mjx_data.replace(qpos=self.mjx_data.qpos.at[0].set(state[0]))
                    forward_reward=(next_state[0]-state[0])/(frame_skip*dt)
                    
                if self.gym_env=="Hopper":
                    forward_reward=next_state[6]
                if self.gym_env=="Walker2d":
                    forward_reward=next_state[9]
                    #state=state[1:]
                elif self.gym_env=="Ant":
                    forward_reward=next_state[13]
                # elif self.gym_env=="Humanoid-v4":
                #     forward_reward=(pos_after - pos_before) / 0.003
            else:
                state = self._dynamics(next_state, action)
                
            done = lax.select(t == self.max_steps, True,False)

            r = self.reward_fn(self.gym_env, next_state, action, forward_reward, self.mjx_data)
            buffer=buffer.append(state, action.flatten(), r.flatten(), done, logp.flatten(), next_state)
           
            mjx_data = jax.tree_util.tree_map(
                lambda x, y: jnp.where(done, x, y), reset_data, mjx_data
            )
            
            next_state = jax.tree_util.tree_map(
                lambda x, y: jnp.where(done, x, y), jnp.concat([mjx_data.qpos,mjx_data.qvel]), next_state
            )
                
            
           

            # then reward calc...
           
            carry = (next_state, key,buffer,mjx_data)
            outputs = (state,next_state, action[0], r, logp,done)
            return carry, outputs
        #pdb.set_trace()
        # Run scan for evaluation_interval steps
        (final_state, _,final_buffer,mjx_data), traj = lax.scan(
            rollout_step,
            (init_state, key,self.buffer,self.mjx_data),
            jnp.arange(self.rollout_length)
        )
        self.buffer=final_buffer
        self.mjx_data=mjx_data
        
        
      
        #states, next_states,actions, rewards, log_ps,dones =traj
        #states, next_states,actions, rewards, log_ps,dones =self.buffer.get()
        # self.buffer.append(states, actions, rewards, dones, log_ps, next_states)
    
        # # reset
        # self.reset()
        return 
   
    def update_ppo(self, states: jnp.ndarray,
                   actions: jnp.ndarray,
                   rewards: jnp.ndarray,
                   dones: jnp.ndarray,
                   log_probs_old: jnp.ndarray,
                   next_states: jnp.ndarray,
                   gamma: float = 0.995,
                   lam: float = 0.97,
                   clip_eps: float = 0.2,
                   vf_coef: float = 1,
                   ent_coef: float = 0.000,
                   num_epochs: int = 10,
                   batch_size: int = 64):

        n_samples = states.shape[0]

        def get_advantages(rewards, values, next_values, dones):
            deltas = rewards + gamma * (1.0 - dones) * next_values - values
            
            adv = []
            gae = 0.0
            for delta, done in zip(deltas[::-1], dones[::-1]):
                gae = delta + gamma * lam * (1.0 - done) * gae
                adv.insert(0, gae)
            return jnp.array(adv)

        if self.value_fn:
            values = self.value_fn.apply_fn({'params': self.value_fn.params}, states)
            next_values = self.value_fn.apply_fn({'params': self.value_fn.params}, next_states)
            values=values.flatten()
            next_values=next_values.flatten()
        else:
            values = jnp.zeros_like(rewards)
            next_values = jnp.zeros_like(rewards)

        advantages = get_advantages(rewards, values, next_values, dones)
       
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def get_minibatches():
            idxs = jnp.arange(n_samples)
            #idxs = jax.random.permutation(jax.random.PRNGKey(0), idxs)
            for start in range(0, n_samples, batch_size):
                if start +batch_size >= n_samples:
                    idxs[start:]
                else:
                    yield idxs[start:start + batch_size]

        def actor_loss_fn(params, minibatch_idxs):
            s = states[minibatch_idxs]
            a = actions[minibatch_idxs]
            old_logp = jnp.array(log_probs_old)[minibatch_idxs]
            adv = advantages[minibatch_idxs]
            #ret = returns[minibatch_idxs]

            mu, log_std = self.policy_net.apply({'params': params}, s)

            logp = self.evaluate_log_pi(s, a,params)
            
            ratio = jnp.exp(logp - old_logp)
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            loss_pi = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
            if self.gym_env in ["CartPole-v1"]:
                entropy = -jnp.mean(logp)
            else:
                entropy_per_sample = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
                entropy = jnp.mean(entropy_per_sample)
           
            loss = loss_pi - ent_coef * entropy
            #print(loss)

            

            return loss
        
        def critic_loss_fn(params, minibatch_idxs):
            s = states[minibatch_idxs]
            ret = returns[minibatch_idxs]

            v = self.value_fn.apply_fn({'params': params}, s).squeeze()
            vf_loss = jnp.mean((ret - v) ** 2)
            loss = vf_coef * vf_loss

            return loss
        mb = jnp.arange(n_samples)
        for _ in range(num_epochs):
            #pdb.set_trace()

            actor_grads = jax.grad(actor_loss_fn)(self.policy_model.params, mb)
            self.policy_model = self.policy_model.apply_gradients(grads=actor_grads)
            if self.value_fn:
                critic_grads = jax.grad(critic_loss_fn)(self.value_fn.params, mb)
                self.value_fn = self.value_fn.apply_gradients(grads=critic_grads)
                
        
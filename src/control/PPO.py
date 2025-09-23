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
from src.control.dynamics import kinematics,kinematics_mujoco,kinematics_mujoco_lax
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
        out = nn.Dense(2 * self.action_dim)(x)   # both μ and logσ
        mu, log_std = jnp.split(out, 2, axis=-1)

        # Clip log_std for numerical stability
        log_std = jnp.clip(log_std, -5, 2)  
        std = jnp.exp(log_std)
        return mu, std
    
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
    def __init__(self,action_dim,mjx_model, dynamics,policy_model,policy_net,args,value_fn=None):
        self.action_dim= action_dim
        self.args = args
        self.gym_env = args.gym_env
        self._dynamics = dynamics # set if needed
        self.mjx_model = mjx_model
        if self.mjx_model is not None :
            self.mjx_data = mjx.make_data(self.mjx_model)
        self.policy_model=policy_model
        self.policy_net=policy_net
        self.value_fn=value_fn
        
    def reset(self):
        """
        Reset the previous action sequence.
        """
        self._previous_action_seq = jnp.zeros(
            (
                1, self.action_dim
            )
        )
    
    def step(self, x, rng):
       mu, std = self.policy_net.apply({'params': self.policy_model.params}, x)
       noise = jax.random.normal(rng, mu.shape)
       action = mu + std * noise
       logp = -0.5 * (((action - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
       logp = jnp.sum(logp, axis=-1)
       return action, logp
   
    
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
            
        

    def generate_session_lax(self, args, D_demo, mpc_method=None, thetas=None):
        key = jax.random.PRNGKey(args.seed)
    
        # Initial state
        init_state = D_demo[0, :args.s_dim]
    
        def rollout_step(carry, t):
            state, key = carry
    
            # RNG split
            key, subkey = jax.random.split(key)
    
            # ---- PPO Policy Action Sampling ----
            mu, std = self.policy_net.apply({'params': self.policy_model.params}, state[None, :])
            noise = jax.random.normal(subkey, mu.shape)
            u = mu + std * noise
            action=jnp.tanh(u)
            # Log prob of u under Gaussian
            logp_u = -0.5 * (((u - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
            logp_u = jnp.sum(logp_u, axis=-1)
            
            # Correction for tanh squashing
            logp = logp_u - jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
            
            state=kinematics_mujoco(self.mjx_model,self.mjx_data,state,action,self._dynamics,self.gym_env)

    
            # ---- Dynamics update ----
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]: 
                if self.gym_env in ["Ant"]:
                    state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
                state=jnp.array(state,dtype=jnp.float64)
                action=jnp.array(action,dtype=jnp.float64)
                if self.gym_env=="Humanoid-v4":
                    pos_before = mass_center(self.mjx_model,state)
                state=kinematics_mujoco(self.mjx_model,self.mjx_data,state.flatten(),action.reshape((1,-1)),self._dynamics,self.gym_env).flatten()
                if self.gym_env=="Humanoid-v4":
                    pos_after = mass_center(self.mjx_model,state)
                if self.gym_env in ["Ant"]:
                    state=state[2:]
                #state=self._dynamics(self.mjx_model,self.mjx_data,state.flatten(), action_seq[0,:].flatten())
                if self.gym_env =="HalfCheetah-v4":
                    #self.mjx_data = self.mjx_data.replace(qpos=self.mjx_data.qpos.at[0].set(state[0]))
                    forward_reward=state[9]
                    
                if self.gym_env=="Hopper":
                    forward_reward=state[6]
                if self.gym_env=="Walker2d":
                    forward_reward=state[9]
                    #state=state[1:]
                elif self.gym_env=="Ant":
                    forward_reward=state[13]
                elif self.gym_env=="Humanoid-v4":
                    forward_reward=(pos_after - pos_before) / 0.003
            else:
                state = self._dynamics(state, action)
    
            # then reward calc...
            r = self.reward_fn(self.gym_env, state, action, forward_reward if "forward_reward" in locals() else None, self.mjx_data)
        
            carry = (state, key)
            outputs = (state, action[0], r, logp)
            return carry, outputs
            
        # Run scan for evaluation_interval steps
        (final_state, _), traj = lax.scan(
            rollout_step,
            (init_state, key),
            jnp.arange(args.evaluation_interval)
        )
    
        states, actions, rewards, logps = traj
    
        # reset
        self.reset()
        return states, jnp.ones((rewards.shape[0],1)), actions, rewards.flatten(), logps.flatten()

    
    
    def generate_session(self, args, D_demo, mpc_method=None, thetas=None):
        
      
        
        states, traj_probs, actions,logps = [], [], [],[]

        key = jax.random.PRNGKey(args.seed)
      
        
        np.random.seed(args.seed)
        env=args.gym_env     
        if env=="CartPole-v1" or env=="Pendulum-v1" or env=="MountainCarContinuous-v0":
            env, env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = env.reset(rng_reset, env_params)
        elif env in["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]:
            self.mjx_data = mjx.make_data(self.mjx_model)
        else:
            env = gym.make(env)
            env_state = env.reset(seed=args.seed)
            
       
        


        state = D_demo[0, :args.s_dim]

       

        #rewards = [true_cost_fn(state)]
        rewards = jnp.zeros((args.evaluation_interval,), dtype=jnp.float32)
        total_rewards=0.0

        pbar = tqdm(total=args.evaluation_interval, desc="Starting")

      
        #self.mjx_data = self.mjx_data.replace(qpos=self.mjx_data.qpos.at[0].set(0))
        for step in range(1, args.evaluation_interval + 1):
            states.append(state)
            _, _, rng_step = jax.random.split(key, 3)
             # ---- PPO Policy Action Sampling ----
            action, logp = self.step( state[None, :], key)
            action=jnp.tanh(action)
            action_seq = action
            logp = jnp.squeeze(logp, axis=0)
            logps.append(logp)
            xposbefore = self.mjx_data.qpos[0]
            self.mjx_data,state,forward_reward=kinematics_mujoco_lax(self.mjx_model,self.mjx_data,action_seq,self._dynamics,self.gym_env)
            xposafter = self.mjx_data.qpos[0]
            
            #action_seq, state_seq = self.forward(state=state,state_train=state_train, gail=args.gail)

        
            
            
            # env_state.theta=state[0]
            # env_state.theta_dot=state[1]
            # env_state.last_u=state[2]
            # _,_, reward, _, _= env.step(
            #     rng_step, env_state, action_seq[0,:], env_params
            # )
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]: 
                if self.gym_env in ["Ant"]:
                    state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
                state=jnp.array(state,dtype=jnp.float64)
                action_seq=jnp.array(action_seq,dtype=jnp.float64)
                if self.gym_env=="Humanoid-v4":
                    pos_before = mass_center(self.mjx_model,state)
                #state=kinematics_mujoco(self.mjx_model,self.mjx_data,state.flatten(),action_seq[0,:].reshape((1,-1)),self._dynamics,self.gym_env).flatten()
                if self.gym_env=="Humanoid-v4":
                    pos_after = mass_center(self.mjx_model,state)
                if self.gym_env in ["Ant"]:
                    state=state[2:]
                #state=self._dynamics(self.mjx_model,self.mjx_data,state.flatten(), action_seq[0,:].flatten())
                if self.gym_env =="HalfCheetah-v4":
                    #self.mjx_data = self.mjx_data.replace(qpos=self.mjx_data.qpos.at[0].set(state[0]))
                    #forward_reward=state[9]
                   # print(forward_reward,self.mjx_data.qvel[0])
                    #print("qpos[0]:", self.mjx_data.qpos[0],
      #"qvel[0]:", self.mjx_data.qvel[0])
                  forward_vel = (xposafter - xposbefore) / (self.mjx_model.opt.timestep)
                  ctrl_cost = 0.1 * np.square(action).sum()
                      
                  reward = forward_vel - ctrl_cost
                  print(forward_reward,forward_vel)
                  #forward_reward=forward_vel
                  
                if self.gym_env=="Hopper":
                    forward_reward=state[6]
                if self.gym_env=="Walker2d":
                    forward_reward=state[9]
                    #state=state[1:]
                elif self.gym_env=="Ant":
                    forward_reward=state[13]
                elif self.gym_env=="Humanoid-v4":
                    forward_reward=(pos_after - pos_before) / 0.003
            else:
                state = self._dynamics(state, action_seq[0,:])  # , reward, terminated, truncated, info = env.step(action_seq_np[0, :])
            state = state.ravel()
            action=action_seq[0,:]
            
            prob = jnp.array([1])

            traj_probs.append(prob.flatten())
            actions.append(action_seq[0].flatten())
           
            if args.gym_env == "CartPole-v1":
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
            if args.gym_env == "Pendulum-v1":
                x=state[0]
                y=state[1]
                theta_pend=jnp.atan2(y,x)
                theta_dot=state[2]
                r = -(jnp.pow(theta_pend,2) + 0.1 * jnp.pow(theta_dot,2) + 0.001 * jnp.pow(action_seq[0,:],2))
               
            if args.gym_env == "MountainCarContinuous-v0":
                r=-0.1 * jnp.pow(action_seq[0,:],2)
                goal_position = 0.45
                goal_velocity = 0.0
                x=state[0]
                xd=state[1]
                if goal_position <= x :
                    r+=100 
            if args.gym_env == "HalfCheetah-v4":
                #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
                ctrl_cost =  0.1 * np.square(action).sum()
                r = forward_reward - ctrl_cost
                r=r.reshape((1,1))
                
            if args.gym_env == "Ant":
                #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
                alive_bonus=1
                if state[1] <0.2 or state[1]>1:
                    alive_bonus=0
                ctrl_cost = 0.5 * np.sum(np.square(action))
                r = forward_reward - ctrl_cost+alive_bonus
                r=r.reshape((1,1))
                
            if args.gym_env == "Hopper":
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
               
                ctrl_cost = 0.001 * np.sum(np.square(action))
                r = forward_reward - ctrl_cost + alive_bonus
                r=r.reshape((1,1))
                
            if args.gym_env == "Walker2d":
                #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
                alive_bonus=1
                if np.abs(state[2])>1 or state[1] <0.8 or state[1]>2:
                    alive_bonus=0
                
                ctrl_cost = 0.001 * np.sum(np.square(action))
                r = forward_reward - ctrl_cost + alive_bonus
                r=r.reshape((1,1))
                
            if args.gym_env == "Humanoid-v4":
                #forward_reward = self.mjx_data.qvel[0]  # usually qvel[0]
                alive_bonus=5
                if state[2] <1 or state[2]>2:
                    alive_bonus=0
                #pdb.set_trace()
                quad_impact_cost = 0.5e-6 * np.square(self.mjx_data.cfrc_ext).sum()
                quad_impact_cost = min(quad_impact_cost, 10)
                ctrl_cost = 0.1 * np.sum(np.square(action))
                r = 1.25*forward_reward - ctrl_cost -quad_impact_cost + alive_bonus
                r=r.reshape((1,1))                
                
                
            #rewards.append(true_cost_fn(state))
            r_scalar = float(np.array(r).reshape(-1)[0])  
            rewards = rewards.at[step-1].set(r_scalar)
            total_rewards+=r_scalar
            #pdb.set_trace()
            pbar.set_description(
                f"  Total True Reward = {total_rewards:.4f} ,Reward True = {r[0].item():.4f}")
                #f"Reward True = {r[0].item():.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            pbar.update(1)
            # probs_fun = jax.vmap(self.predict_probs, (0, None, 0))
            # prob = self.predict_probs(action_seq[0],
            #                           self.covariance[0],
            #                           ction_seq)

            
        #pdb.set_trace()
        #rewards = jnp.array(rewards)
        pbar.close()

        self.reset()

        
        return states, traj_probs, actions,rewards,logps

   
    def update_ppo(self, states: jnp.ndarray,
                   actions: jnp.ndarray,
                   rewards: jnp.ndarray,
                   dones: jnp.ndarray,
                   log_probs_old: jnp.ndarray,
                   next_states: jnp.ndarray,
                   gamma: float = 0.995,
                   lam: float = 0.97,
                   clip_eps: float = 0.2,
                   vf_coef: float = 0.5,
                   ent_coef: float = 0.000,
                   num_epochs: int = 10,
                   batch_size: int = 2048):

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
            idxs = jax.random.permutation(jax.random.PRNGKey(0), idxs)
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

            mu, std = self.policy_net.apply({'params': params}, s)
            #dist = mu + std * jax.random.normal(jax.random.PRNGKey(0), mu.shape)
            #logp = -0.5 * (((a - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
            #logp = jnp.sum(logp, axis=-1)
            # Invert tanh: recover pre-tanh u from squashed action a
            u = jnp.arctanh(jnp.clip(a, -0.999999, 0.999999))  # avoid NaN near ±1
        
            # Gaussian log prob of u
            logp_u = -0.5 * (((u - mu) / std) ** 2 +
                             2 * jnp.log(std) +
                             jnp.log(2 * jnp.pi))
            logp_u = jnp.sum(logp_u, axis=-1)
        
            # Change-of-variables correction (Jacobian of tanh)
            logp = logp_u - jnp.sum(jnp.log(1 - a**2 + 1e-6), axis=-1)

            ratio = jnp.exp(logp - old_logp)
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            loss_pi = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
            if self.gym_env in ["CartPole-v1"]:
                entropy = -jnp.mean(logp)
            else:
                entropy_per_sample = jnp.sum(jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
                entropy = jnp.mean(entropy_per_sample)
           
            loss = loss_pi - ent_coef * entropy

            

            return loss
        
        def critic_loss_fn(params, minibatch_idxs):
            s = states[minibatch_idxs]
            ret = returns[minibatch_idxs]

            v = self.value_fn.apply_fn({'params': params}, s).squeeze()
            vf_loss = jnp.mean((ret - v) ** 2)
            loss = vf_coef * vf_loss

            return loss

        for _ in range(num_epochs):
            for mb in get_minibatches():
                actor_grads = jax.grad(actor_loss_fn)(self.policy_model.params, mb)
                self.policy_model = self.policy_model.apply_gradients(grads=actor_grads)
                if self.value_fn:
                    critic_grads = jax.grad(critic_loss_fn)(self.value_fn.params, mb)
                    self.value_fn = self.value_fn.apply_gradients(grads=critic_grads)
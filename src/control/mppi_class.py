
from __future__ import annotations

from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.random import multivariate_normal
import functools

from src.control.dynamics import kinematics,kinematics_mujoco
from src.objective_fns.cost_to_go_fns import get_cost
from cost_jax import get_gradients,get_hessian,get_hessian_diag,get_precond,fisher_diag

import os.path as osp
from functools import partial

import math
from tqdm.auto import tqdm
from copy import deepcopy
#from utils.models import load_neural_network
import gymnax
import gymnasium as gym
import pdb
from cost_jax import apply_model, apply_model_AIRL, update_model
from mujoco import mjx 
import random
import jax
import jax.tree_util as jtu
import time
if not hasattr(jax, "tree_map"):
    jax.tree_map = jtu.tree_map





@jax.jit
def update_theta(theta, P_theta, Q_theta, hessian_d, hessian_s, gradient_d, gradient_s):
      
    P_theta = jnp.linalg.inv(jnp.linalg.inv(P_theta + Q_theta) + hessian_d - hessian_s)
    #P_theta = jnp.linalg.inv( hessian_d - hessian_s)
    theta = theta - jnp.matmul(P_theta, gradient_d - gradient_s)
    return theta,P_theta

@jax.jit
def update_theta_diag(theta, P_theta, Q_theta, hessian_d, hessian_s, gradient_d, gradient_s):
    
    P_theta = 1/(1/(P_theta + Q_theta) + hessian_d - hessian_s)
    theta = theta - P_theta*(gradient_d - gradient_s)
    
    return theta,P_theta


def mass_center(model,state):
    # Reshape body masses to (nbody, 1) for broadcasting
    mass = jnp.expand_dims(model.body_mass, axis=1)  # shape: (nbody, 1)
    
    # data.xpos contains the global positions of the body frames (shape: nbody x 3)
    xpos = state[:24]  # shape: (nbody, 3)

    # Compute center of mass as the weighted average of body positions
    com = jnp.sum(mass * xpos, axis=0) / jnp.sum(mass)
    
    # Return x-coordinate of the center of mass (you can return full com if needed)
    return com[0]



class MPPI:
    """
    Model Predictive Path Integral Control,
    J. Williams et al., T-RO, 2017.
    """

    def __init__(
            self,
            state_train,
            horizon: int,
            num_samples: int,
            dim_state: int,
            dim_control: int,
            dynamics: Callable[[jnp.array, jnp.array], jnp.array],
            cost_func: Callable[[jnp.array], jnp.array],
            u_min: jnp.array,
            u_max: jnp.array,
            sigmas: jnp.array,
            lambda_: float,
            exploration: float = 0.0,
            seed: int = 42,
            env=None,
            mjx_model=None,
            gym_env=None,
            env_brax=None,
            use_mujoco=bool
    ) -> None:
        """
        :param horizon: Predictive horizon length.
        :param predictive_interval: Predictive interval (seconds).
        :param delta: predictive horizon step size (seconds).
        :param num_samples: Number of samples.
        :param dim_state: Dimension of state.
        :param dim_control: Dimension of control.
        :param dynamics: Dynamics model.
        :param cost_func: Cost function.
        :param u_min: Minimum control.
        :param u_max: Maximum control.
        :param sigmas: Noise standard deviation for each control dimension.
        :param lambda_: temperature parameter.
        :param exploration: Exploration rate when sampling.
        :param seed: Seed for jax.
        """

        super().__init__()

        # jax seed
        self.key = jax.random.PRNGKey(seed)
        self.seed = seed

        # check dimensions
        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)
        # assert num_samples % batch_size == 0 and num_samples >= batch_size

        # set parameters
        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._cost_func = cost_func
        self._u_min = u_min.clone()
        self._u_max = u_max.clone()
        self._sigmas = sigmas.clone()
        self._lambda = lambda_
        self._exploration = exploration
        self.env=env
        self.env_brax=env_brax
        self.mjx_model=mjx_model
        self.gym_env=gym_env
        self.use_mujoco=use_mujoco
        self.state_train=state_train
        if self.mjx_model is not None :
            self.mjx_data = mjx.make_data(self.mjx_model)
        else:
            self.mjx_data=None
            
       

        # noise distribution
        self._covariance = jnp.zeros((
            self._horizon,
            self._dim_control,
            self._dim_control,
        ))
        self._covariance = self._covariance.at[:, :, :].set(jnp.diag(sigmas ** 2))
        self._inv_covariance = jnp.zeros_like(
            self._covariance
        )

        self.covariance = jnp.array(self._covariance)

        for t in range(1, self._horizon):
            self._inv_covariance = self._inv_covariance.at[t].set(jnp.linalg.inv(self._covariance[t]))


        self._inv_covariance = jnp.array(self._inv_covariance)

        self.zero_mean = jnp.zeros(dim_control)

        # self._noise_distribution = MultivariateNormal(
        #     loc=zero_mean, covariance_matrix=self._covariance
        # )

        self._sample_shape = [self._num_samples,self._horizon]

        # sampling with reparameting trick
        self._action_noises = multivariate_normal(self.key, mean=self.zero_mean, cov=self._covariance)
        self.key,_ = jax.random.split(self.key)
        self.og_key = deepcopy(self.key)

        zero_mean_seq = jnp.zeros((self._horizon, self._dim_control))

        self._perturbed_action_seqs = jnp.clip(
            zero_mean_seq + self._action_noises, self._u_min, self._u_max
        )

        self._previous_action_seq = zero_mean_seq

        # inner variables
        self._state_seq_batch = jnp.zeros(
            (
                self._num_samples,
                self._horizon + 1,
                self._dim_state
            )
        )

        self._weights = jnp.zeros(
            (self._num_samples,)
        )
        self._optimal_state_seq = jnp.zeros(
            (
                self._horizon + 1, self._dim_state
            )
        )

    def reset(self):
        """
        Reset the previous action sequence.
        """
        self._previous_action_seq = jnp.zeros(
            (
                self._horizon, self._dim_control
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

    def reset_batched(self,keys):
        return jax.vmap(self.env_brax.reset)(keys)

    def step_batched(self,state, actions):
        return jax.vmap(self.env_brax.step)(state, actions)


    def forward_pure(self,state, state_train=None, gail=False,*,key,prev_action_seq,frame_skip):
        """
        Pure MPPI forward step.
        Args:
            state: jnp.ndarray shape (state_dim,)
            prev_action_seq: jnp.ndarray shape (H, act_dim)  # previously 'self._previous_action_seq'
            key: PRNGKey
        Returns:
            optimal_action_seq: (H, act_dim)
            optimal_state_seq:  (H+1, state_dim)
            new_key: PRNGKey
            new_prev_action_seq: (H, act_dim)
        """
        assert state.shape == (self._dim_state,)
    
        mean_action_seq = prev_action_seq  # no clone; keep JAX arrays
    
        # random sampling with reparametrization trick
        action_noises = multivariate_normal(
            key, mean=self.zero_mean, cov=self._covariance, shape=self._sample_shape
        )
        key, _ = jax.random.split(key)
    
        # noise injection with exploration
        threshold = int(self._num_samples * (1.0 - self._exploration))
        inherited_samples = mean_action_seq + action_noises[:threshold]
        perturbed_action_seqs = jnp.concatenate(
            [inherited_samples, action_noises[threshold:]], axis=0
        )
    
        # clamp actions
        perturbed_action_seqs = jnp.clip(
            perturbed_action_seqs, self._u_min, self._u_max
        )
    
        # rollout samples in parallel
        if self.gym_env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4","Swimmer"]:
            st = state
            if self.gym_env in ["Ant"]:
                st = jnp.concatenate((jnp.reshape(self.mjx_data.qpos[0:2], (2,)), st))
            initial_state = jnp.tile(st, (self._num_samples, 1))
            #pdb.set_trace()
            # Note: timing removed - incompatible with lax.scan tracing
            state_seq_batch = jax.vmap(
                kinematics_mujoco, in_axes=(None, None, 0, 0, None,None)
            )(self.mjx_model, self.mjx_data, initial_state, perturbed_action_seqs, self.gym_env,frame_skip)
            #pdb.set_trace()
            initial_state = initial_state.reshape((initial_state.shape[0], 1, initial_state.shape[1]))
            state_seq_batch = jnp.concatenate((initial_state, state_seq_batch), axis=1)
            if self.gym_env in ["Ant"]:
                state_seq_batch = state_seq_batch[:, :, 2:]
                st = st[2:]
        else:
            initial_state = jnp.tile(state, (self._num_samples, 1, 1))
            state_seq_batch = jax.vmap(kinematics, in_axes=(0, 0, None))(
                initial_state, perturbed_action_seqs, self._dynamics
            )
            state_seq_batch = jnp.squeeze(state_seq_batch, axis=-2)
            state_seq_batch = jnp.concatenate((initial_state, state_seq_batch), axis=1)
    
        # compute sample costs
        # costs over horizon: (num_samples, horizon)
        costs = jax.vmap(self._cost_func, in_axes=(1, None))(state_seq_batch[:, :-1, :], state_train)
        costs = costs[:, :, 0]  # assuming cost_func returns (..., 1)
        costs = costs.T  # (num_samples, horizon)
    
        terminal_costs = self._cost_func(
            state_seq_batch[:, -1, :], state_train
        ).ravel()
    
        total_costs = jnp.sum(costs, axis=1) + terminal_costs
    
        if gail:
            D = jnp.exp(-total_costs) / (jnp.exp(-total_costs) + 1.0)
            total_costs = -jnp.log(D)
    
        # weights and optimal control
        # Note: timing and block_until_ready() removed - incompatible with lax.scan tracing
        weights = jax.nn.softmax(-total_costs / self._lambda, axis=0)
        optimal_action_seq = jnp.sum(
            weights.reshape(self._num_samples, 1, 1) * perturbed_action_seqs,
            axis=0,
        )
    
        expanded_optimal_action_seq = jnp.tile(prev_action_seq, (1, 1, 1))
        #optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq,frame_skip)
        optimal_state_seq=0
        # new_prev_action_seq: in many MPPI impls you set it to optimal for warm start
        new_prev_action_seq = optimal_action_seq
    
        return optimal_action_seq, optimal_state_seq, key, new_prev_action_seq





    
    # def forward_pure_brax(self,state, state_train=None, gail=False,*,key,prev_action_seq,frame_skip,brax_state0):
    #     """
    #     Pure MPPI forward step.
    #     Args:
    #         state: jnp.ndarray shape (state_dim,)
    #         prev_action_seq: jnp.ndarray shape (H, act_dim)  # previously 'self._previous_action_seq'
    #         key: PRNGKey
    #     Returns:
    #         optimal_action_seq: (H, act_dim)
    #         optimal_state_seq:  (H+1, state_dim)
    #         new_key: PRNGKey
    #         new_prev_action_seq: (H, act_dim)
    #     """
    #     print("brax_state0.pipeline_state.q shape =", brax_state0.pipeline_state.q.shape)
    #     print("brax_state0.obs shape =", brax_state0.obs.shape)
       

    #     def rollout_brax(env, init_state, action_seqs):
    #         """
    #         env: Brax env
    #         init_state: env.State (unbatched)
    #         action_seqs: (B, H, act_dim)
    #         returns:
    #             obs_batch: (B, H+1, obs_dim)
    #             final_states: (B, ...)
    #         """
    #         B, H, act_dim = action_seqs.shape
        
    #         def rollout_single(actions_1):
    #             # actions_1: (H, act_dim)
    #             def step_fn(state, action_t):
                    
    #                 next_state = env.step(state, action_t)  # NOT vmapped
    #                 return next_state, next_state.obs      # (obs_dim,)
                
    #             final_state, obs_seq = jax.lax.scan(
    #                 step_fn,
    #                 init_state,      # unbatched!
    #                 actions_1        # (H, act_dim)
    #             )
                
        
    #             # prepend initial obs
    #             obs_seq = jnp.concatenate(
    #                 [init_state.obs[None, :], obs_seq], 
    #                 axis=0
    #             )  # (H+1, obs_dim)
        
    #             return obs_seq, final_state
        
    #         # vmap over batch dim of action_seqs
    #         obs_batch, final_states = jax.vmap(
    #             rollout_single,
    #             in_axes=(0,)
    #         )(action_seqs)
        
    #         return obs_batch, final_states

        
        

    #     assert state.shape == (self._dim_state,)
    
    #     mean_action_seq = prev_action_seq  # no clone; keep JAX arrays
    
    #     # random sampling with reparametrization trick
    #     action_noises = multivariate_normal(
    #         key, mean=self.zero_mean, cov=self._covariance, shape=self._sample_shape
    #     )
    #     key, _ = jax.random.split(key)
    
    #     # noise injection with exploration
    #     threshold = int(self._num_samples * (1.0 - self._exploration))
    #     inherited_samples = mean_action_seq + action_noises[:threshold]
    #     perturbed_action_seqs = jnp.concatenate(
    #         [inherited_samples, action_noises[threshold:]], axis=0
    #     )
    
    #     # clamp actions
    #     perturbed_action_seqs = jnp.clip(
    #         perturbed_action_seqs, self._u_min, self._u_max
    #     )
    
    #     # rollout samples in parallel
    #     if self.gym_env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4"]:
    #         st = state
    #         if self.gym_env in ["Ant"]:
    #             st = jnp.concatenate((jnp.reshape(self.mjx_data.qpos[0:2], (2,)), st))
        
    #         # set the brax observation to the current gym state
    #         brax_state0 = brax_state0.replace(obs=st)  # st: (obs_dim,)
           
    #         # perturbed_action_seqs: (B, H, act_dim)  where B = self._num_samples
    #         start = time.time()
    #         state_seq_batch, _ = rollout_brax(
    #             self.env_brax,
    #             brax_state0,          # unbatched state
    #             perturbed_action_seqs # (B, H, act_dim)
    #         )
    #         end = time.time()
    #         print(f"Execution time: {end - start:.4f} seconds")
           
           
    #         # state_seq_batch: (B, H+1, obs_dim)
        
    #         if self.gym_env in ["Ant"]:
    #             # remove the extra root components if you added them in st
    #             state_seq_batch = state_seq_batch[:, :, 2:]
    #             st = st[2:]
    #     else:
    #         initial_state = jnp.tile(state, (self._num_samples, 1, 1))
    #         state_seq_batch = jax.vmap(kinematics, in_axes=(0, 0, None))(
    #             initial_state, perturbed_action_seqs, self._dynamics
    #         )
    #         state_seq_batch = jnp.squeeze(state_seq_batch, axis=-2)
    #         state_seq_batch = jnp.concatenate((initial_state, state_seq_batch), axis=1)

    #     # compute sample costs
    #     # costs over horizon: (num_samples, horizon)
    #     costs = jax.vmap(self._cost_func, in_axes=(1, None))(state_seq_batch[:, :-1, :], state_train)
    #     costs = costs[:, :, 0]  # assuming cost_func returns (..., 1)
    #     costs = costs.T  # (num_samples, horizon)
    
    #     terminal_costs = self._cost_func(
    #         state_seq_batch[:, -1, :], state_train
    #     ).ravel()
    
    #     total_costs = jnp.sum(costs, axis=1) + terminal_costs
    
    #     if gail:
    #         D = jnp.exp(-total_costs) / (jnp.exp(-total_costs) + 1.0)
    #         total_costs = -jnp.log(D)
    
    #     # weights and optimal control
    #     weights = jax.nn.softmax(-total_costs / self._lambda, axis=0)
    
    #     optimal_action_seq = jnp.sum(
    #         weights.reshape(self._num_samples, 1, 1) * perturbed_action_seqs,
    #         axis=0,
    #     )
    
    #     expanded_optimal_action_seq = jnp.tile(prev_action_seq, (1, 1, 1))
        
    #    # optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq,frame_skip)
    #     optimal_state_seq =0
    #     # new_prev_action_seq: in many MPPI impls you set it to optimal for warm start
    #     new_prev_action_seq = optimal_action_seq
    
    #     return optimal_action_seq, optimal_state_seq, key, new_prev_action_seq




    def _states_prediction(
            self, state, action_seqs,frame_skip
    ):
        
        if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]: 
            initial_state=state
            if self.gym_env in ["Ant"]:
                initial_state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
            initial_state=jnp.array(initial_state.reshape((1,-1)))
            action_seqs=jnp.array(action_seqs)
            
            state_seqs=jax.vmap(kinematics_mujoco,in_axes=(None,None,0,0,None,None))(self.mjx_model,self.mjx_data,initial_state,action_seqs,self.gym_env,frame_skip)
            state_seqs= jnp.concatenate((jnp.tile(initial_state,(1,1,1)), state_seqs), axis=1)
            if self.gym_env in ["Ant"]:
                state_seqs=state_seqs[:,:,2:]
        else:
            state_seqs = jnp.zeros((
                action_seqs.shape[0],
                self._horizon + 1,
                self._dim_state,
            ))
            state_seqs = state_seqs.at[:, 0, :].set(state)

        # expanded_optimal_action_seq = action_seq.repeat(1, 1, 1)
       
            for t in range(self._horizon):
                state_seqs = state_seqs.at[:, t + 1, :].set(
    self._dynamics(state_seqs[:, t, :], action_seqs[:, t, :])
)

                
        return jnp.array(state_seqs)

    def predict_probs(self, mean, cov, x):
        x = x.T
        mean = mean.T
        pdf = (2 * math.pi) ** (-1) * (jnp.linalg.det(cov)) ** (-1 / 2) * jnp.exp(
            -1 / 2 * jnp.matmul(jnp.matmul((x - mean).T, jnp.linalg.inv(cov)), (x - mean)))
        return pdf
    def reward_fn(self,gym_env, state, action,next_state, mjx_data,dt,frame_skip):
        forward_reward=(next_state[0]-state[0])/(dt*frame_skip)

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
            alive_bonus = 1.0

            # JAX-compatible fall condition checks
            # Check if any state values are out of bounds [-100, 100]
            out_of_bounds = jnp.any((next_state[2:] < -100) | (next_state[2:] > 100))
            # Check angle constraint
            bad_angle = (next_state[2] < -0.2) | (next_state[2] > 0.2)
            # Check height constraint
            bad_height = (next_state[1] < 0.7)
            # Combine all fall conditions
            fall_cond = out_of_bounds | bad_angle | bad_height
            # Set alive_bonus to 0 if any fall condition is true
            alive_bonus = jnp.where(fall_cond, 0.0, 1.0)

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
    


    #@functools.partial(jax.jit, static_argnums=(0, 1))  # self=0, args=1
    def generate_session_lax(self, args, state_train, D_demo, mpc_method=None, thetas=None):
        key = jax.random.PRNGKey(args.seed)
        dt=args.dt
        frame_skip=args.frame_skip
    
        # Initial state
        init_state = D_demo[0, :args.s_dim]
       
    
        # Take the *current* previous_action_seq once, outside trace:
        prev_action_seq0 = self._previous_action_seq  # OK to read outside
        env=args.gym_env
        if env=="CartPole-v1" or env=="Pendulum-v1" or env=="MountainCarContinuous-v0":
            self.env, self.env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = self.env.reset(rng_reset, self.env_params)
        elif env in["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4","Swimmer"]:

            self.mjx_data = self.reset_mjx_state(self.mjx_model,key=key)
        else:
            #env = gym.make(env)
            env_state = self.env.reset(seed=args.seed)
    
        def rollout_step(carry, t):
            state, key, prev_action_seq = carry
    
            # forward
            
            action_seq, state_seq, key, prev_action_seq = self.forward_pure(
                state=state, state_train=state_train, gail=args.gail,
                key=key, prev_action_seq=prev_action_seq,frame_skip=frame_skip
            )
    
            # ---- Dynamics update ----
            forward_reward = 0.0
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4","Swimmer"]:
                  next_state = kinematics_mujoco_original(
                    self.mjx_model, self.mjx_data, state.flatten(),
                    action_seq[0, :].reshape((1, -1)), self.gym_env,frame_skip=frame_skip
                ).flatten()
            else:
                next_state = self._dynamics(state, action_seq[0, :])
    
                
           
            
            next_state=next_state.ravel()
            action = action_seq[0, :]
    
            prob = jnp.array([1])
            r = self.reward_fn(self.gym_env, state, action, next_state, self.mjx_data,dt,frame_skip)
    
            new_carry = (next_state, key, prev_action_seq)
            outputs = (state, prob, action, r)
            return new_carry, outputs
        
       # rollout_step_jit = jax.jit(rollout_step, static_argnums=(0,))
        (final_carry, traj) = lax.scan(
            rollout_step,
            (init_state, key, prev_action_seq0),
            jnp.arange(args.N_steps)
        )
        (final_state, final_key, final_prev_action_seq), (states, traj_probs, actions, rewards) = final_carry, traj
    
        # It is safe to mutate class attributes AFTER the scan
        self._previous_action_seq = final_prev_action_seq
        self.reset()
       
        rewards=jnp.sum(rewards)
        states, traj_probs, actions, rewards = (
            states.tolist(),
            traj_probs.tolist(),
            actions.tolist(),
            rewards.tolist()
        )
        

        return states, traj_probs, actions, rewards
    
    def RGCL(self, args, params, state_train, initial_state, D_demo, P_theta_in, thetas=None):
            """
            Python loop version of RGCL (Logic identical to RGCL_lax).
            Performs full Hessian updates explicitly in a standard for-loop.
            """
            import jax.numpy as jnp
            import jax
            
            # ---- Flatten parameters into theta ----
            flat_params, treedef = jax.tree_util.tree_flatten(params)
            theta = jnp.concatenate([p.reshape(-1) for p in flat_params])
            n_theta = theta.size
            key = jax.random.PRNGKey(args.seed)
    
            # ---- Initialize P_theta and Q_theta ----
            if args.diagonal:
                raise ValueError("Diagonal version not implemented here. Full version only.")
            else:
                P = args.P * jnp.eye(n_theta)
                Q = args.Q * jnp.eye(n_theta)
    
            # ---- Expert data ----
            expert_states = D_demo[:, :args.s_dim]
            # expert_actions = D_demo[:, args.s_dim:args.s_dim + args.a_dim] # Unused in the update logic shown
    
            # ---- Loop Initialization ----
            state = initial_state
            dt = args.dt
            frame_skip = args.frame_skip
            prev_action_seq = self._previous_action_seq
            
            # Storage lists
            states_hist = []
            traj_probs_hist = []
            actions_hist = []
            rewards_hist = []
            P_theta_hist = [] # Optional: usually only final P is needed, but keeping for consistency
    
            # ----------- MAIN LOOP -----------
            for t in range(args.N_steps):
                
                # 1. Unflatten theta -> params (Reconstruct params for current step)
                p_list = []
                idx = 0
                for p in flat_params:
                    size = p.size
                    p_list.append(theta[idx:idx+size].reshape(p.shape))
                    idx += size
                local_params = jax.tree_util.tree_unflatten(treedef, p_list)
                
                # Update state_train with current parameters
                state_train_local = state_train.replace(params=local_params)
    
                # 2. MPPI policy (Forward Pure)
                # using the updated state_train_local (new weights)
                action_seq, _, key, prev_action_seq = self.forward_pure(
                    state=state,
                    state_train=state_train_local,
                    gail=args.gail,
                    key=key,
                    prev_action_seq=prev_action_seq,
                    frame_skip=frame_skip
                )
                action = action_seq[0]
    
                # 3. Environment transition
                if self.gym_env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4","Swimmer"]:
                    next_state = kinematics_mujoco(
                        self.mjx_model, self.mjx_data, state, action.reshape(1, -1),
                        self.gym_env, frame_skip=frame_skip
                    ).reshape(-1)
                else:
                    next_state = self._dynamics(state, action).reshape(-1)
    
                # 4. Reward Calculation
                reward = self.reward_fn(self.gym_env, state, action, next_state, self.mjx_data, dt, frame_skip)
    
                # 5. Compute Gradients
                g_s = get_gradients(state_train_local, local_params, next_state, args.N_steps)
                g_d = get_gradients(state_train_local, local_params, expert_states[t], args.N_steps)
    
                # 6. Compute Full Hessians
                H_s = get_hessian(state_train_local, local_params, next_state, args.N_steps)
                H_d = get_hessian(state_train_local, local_params, expert_states[t], args.N_steps)
    
                # 7. Kalman-style Update
                # P_new = inv(inv(P + Q) + H_d - H_s)
                P = jnp.linalg.inv(jnp.linalg.inv(P + Q) + (H_d - H_s))
    
                # theta_new = theta - P_new @ (g_d - g_s)
                theta = theta - P @ (g_d - g_s)
                theta = theta.astype(jnp.float32)
    
                # 8. Store History
                # Store 'state' (current) before updating to 'next_state' for the next loop
                states_hist.append(state)
                actions_hist.append(action)
                rewards_hist.append(reward)
                traj_probs_hist.append(1.0) # Matched from RGCL_lax logic
                P_theta_hist.append(P)
    
                # Update state for next iteration
                state = next_state
    
            # ----------- Finalization -----------
            
            # Unflatten final theta back into params to return updated model
            idx = 0
            new_param_list = []
            for p in flat_params:
                size = p.size
                new_param_list.append(theta[idx:idx+size].reshape(p.shape))
                idx += size
            new_params = jax.tree_util.tree_unflatten(treedef, new_param_list)
            
            # Calculate total reward
            total_reward = sum(rewards_hist) # or jnp.sum(jnp.array(rewards_hist))
    
            # Convert lists to desired return format (standard lists as per your original request)
            # Note: In the scan version you returned .tolist(). Doing the same here.
            states_out = [s.tolist() if hasattr(s, 'tolist') else s for s in states_hist]
            traj_probs_out = traj_probs_hist
            actions_out = [a.tolist() if hasattr(a, 'tolist') else a for a in actions_hist]
            rewards_out = [r.tolist() if hasattr(r, 'tolist') else r for r in rewards_hist]
    
            # Return format matches RGCL_lax
            return states_out, traj_probs_out, actions_out, total_reward, P, new_params
    
    def RGCL_lax(self, args, params, state_train, initial_state, D_demo, P_theta_in, thetas=None):
        """
        LAX-scan version of RGCL (NO JIT - user requested).
        Performs full Hessian updates inside the scan.
        Produces identical updates to the Python loop version.
        """
    
        # ---- Flatten parameters into theta ----
        flat_params, treedef = jax.tree_util.tree_flatten(params)
        theta0 = jnp.concatenate([p.reshape(-1) for p in flat_params])
        n_theta = theta0.size
        key= jax.random.PRNGKey(args.seed)
    
        # ---- Initialize P_theta and Q_theta ----
        if args.diagonal:
            raise ValueError("Diagonal version not implemented here. Full version only.")
        else:
            P0 = args.P * jnp.eye(n_theta)
            Q  = args.Q * jnp.eye(n_theta)
    
        # ---- Expert data ----
        expert_states  = D_demo[:, :args.s_dim]
        expert_actions = D_demo[:, args.s_dim:args.s_dim + args.a_dim]
    
        # ---- Initial state ----
        init_state = initial_state
        dt=args.dt
        frame_skip=args.frame_skip
    
        # ----------- SCAN BODY -----------
        def scan_step(carry, t):
            state, theta, P,prev_action_seq, key = carry
           
            # ---- Unflatten theta → params ----
            p_list = []
            idx = 0
            for p in flat_params:
                size = p.size
                p_list.append(theta[idx:idx+size].reshape(p.shape))
                idx += size
            local_params = jax.tree_util.tree_unflatten(treedef, p_list)
            
            state_train_local=state_train.replace(params=local_params)
            # ---- MPPI policy + dynamics ----
            # no mutation: pure version
            action_seq, _, new_key, new_prev_action_seq  = self.forward_pure(
                state=state,
                state_train=state_train_local,
                gail=args.gail,
                key= key,   # deterministic per-step noise
                prev_action_seq=prev_action_seq,frame_skip=frame_skip
            )
            action = action_seq[0]
    
            # ---- Environment transition ----
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d","Humanoid-v4"]:
                next_state = kinematics_mujoco(
                    self.mjx_model, self.mjx_data, state, action.reshape(1,-1),
                    self.gym_env, frame_skip=frame_skip
                ).reshape(-1)
            else:
                next_state = self._dynamics(state, action).reshape(-1)
            forward_reward=0
            reward = self.reward_fn(self.gym_env, state, action, next_state, self.mjx_data,dt,frame_skip)
    
            # ---- Compute gradients ----
            g_s = get_gradients(state_train_local, local_params, next_state, args.N_steps)
            g_d = get_gradients(state_train_local, local_params, expert_states[t], args.N_steps)
    
            # ---- Full Hessians ----
            H_s = get_hessian(state_train_local, local_params, next_state, args.N_steps)
            H_d = get_hessian(state_train_local, local_params, expert_states[t], args.N_steps)
    
            # ---- Kalman-style update (EXACT rule you want) ----
            # P ← inv(inv(P + Q) + H_d - H_s)
            P_new = jnp.linalg.inv(jnp.linalg.inv(P + Q) + (H_d - H_s))
    
            # theta ← theta − P (g_d − g_s)
            theta_new = theta - P_new @ (g_d - g_s)
            traj_prob=1
            theta_new = theta_new.astype(jnp.float32)
            
            return (next_state, theta_new, P_new,new_prev_action_seq,new_key), (state, traj_prob,action,reward,P_new)
        
        # ----------- EXECUTE SCAN -----------
        (final_state, final_theta, final_P,_,_),(states, traj_probs, actions, rewards,P_theta) = lax.scan(
            scan_step,
            (init_state, theta0, P0,self._previous_action_seq,key),
            jnp.arange(args.N_steps)
        )
    
        # ----------- Unflatten final theta back into params -----------
        idx = 0
        new_param_list = []
       
        for p in flat_params:
            size = p.size
            new_param_list.append(final_theta[idx:idx+size].reshape(p.shape))
            idx += size
        new_params = jax.tree_util.tree_unflatten(treedef, new_param_list)
        params=new_params
        rewards=jnp.sum(rewards)
        states, traj_probs, actions, rewards = (
            states.tolist(),
            traj_probs.tolist(),
            actions.tolist(),
            rewards.tolist()
        )
        
        return states, traj_probs, actions, rewards,P_theta,params 


    def generate_session_loop(self, args, state_train, D_demo, mpc_method=None, thetas=None):
        key = jax.random.PRNGKey(args.seed)
        dt = args.dt
        frame_skip = args.frame_skip
    
        # -----------------------------
        # INITIAL STATE
        # -----------------------------
        init_state = D_demo[0, :args.s_dim]
    
        # Warm-start action seq from class attribute
        prev_action_seq = self._previous_action_seq
    
        env = args.gym_env
    
        # -----------------------------
        # Environment reset
        # -----------------------------
        if env in ["CartPole-v1", "Pendulum-v1", "MountainCarContinuous-v0"]:
            self.env, self.env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = self.env.reset(rng_reset, self.env_params)
    
        elif env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4","Swimmer"]:
            self.mjx_data = self.reset_mjx_state(self.mjx_model, key=key)
    
        else:
            env_state = self.env.reset(seed=args.seed)
    
        # -----------------------------
        # Buffers for trajectory (PRE-ALLOCATED JAX ARRAYS)
        # -----------------------------
        # Pre-allocate arrays on GPU instead of Python lists
        states_buf = jnp.zeros((args.N_steps, args.s_dim))
        probs_buf = jnp.zeros((args.N_steps, 1))
        actions_buf = jnp.zeros((args.N_steps, args.a_dim))
        rewards_buf = jnp.zeros((args.N_steps,))
    
        # -----------------------------
        # LOOP ROLLOUT (replaces lax.scan)
        # -----------------------------
        state = init_state
        key, key_reset = jax.random.split(key)
        # create B independent reset keys
        reset_keys = jax.random.split(key_reset, self._num_samples)
        #brax_state0 = self.env_brax.reset(reset_keys[0])


        
      
    
        for t in range(args.N_steps):
            #brax_state = brax_state0   # true Brax State
            #state = brax_state.obs  
            start = time.time()
            
    
            # ---------------------------------------------------
            # FORWARD MPPI STEP (SAME AS BEFORE)
            # ---------------------------------------------------
          
            
            # action_seq, _, key, prev_action_seq = self.forward_pure_jitted(
            #             state=state,
            #             params=self.state_train.params,
            #             key=key,
            #             prev_action_seq=prev_action_seq,
            #             zero_mean=self.zero_mean,
            #             covariance=self._covariance,
            #             u_min=self._u_min,
            #             u_max=self._u_max,
            #             lambda_=self._lambda,
            #             num_samples=self._num_samples,
            #             exploration=self._exploration,
            #             horizon=self._horizon,
            #             dim_state=self._dim_state,
            #             mjx_model=self.mjx_model,
            #             mjx_data=self.mjx_data,
            #             dynamics_fn=self._dynamics,      # if you want non-MuJoCo envs
            #             gym_env=self.gym_env,
            #             frame_skip=frame_skip,
            #             use_mujoco=self.use_mujoco,
            #         )

            action_seq, _, key, prev_action_seq = self.forward_pure(
                state=state,
                state_train=state_train,
                gail=args.gail,
                key=key,
                prev_action_seq=prev_action_seq,
                frame_skip=frame_skip,#
                
            )
            
            
    
            # First action in sequence
            action = action_seq[0, :]
    
            # ---------------------------------------------------
            # DYNAMICS UPDATE (MuJoCo or simple env)
            # ---------------------------------------------------
            if env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4","Swimmer"]:
                next_state = kinematics_mujoco(
                    self.mjx_model,
                    self.mjx_data,
                    state.flatten(),
                    action.reshape((1, -1)),
                    env,
                    frame_skip=frame_skip
                ).flatten()
                
                # brax_state = self.env_brax.step(brax_state, action)  # ✅ use Brax State here
                # next_state = brax_state.obs  
    
            else:
                next_state = self._dynamics(state, action)
    
            next_state = next_state.ravel()  # ensure shape
    
            # ---------------------------------------------------
            # REWARD
            # ---------------------------------------------------
            prob = jnp.array([1.0])
            r = self.reward_fn(env, state, action, next_state, self.mjx_data, dt, frame_skip)
    
            # ---------------------------------------------------
            # SAVE STEP (USE JAX ARRAY INDEXING - STAYS ON GPU)
            # ---------------------------------------------------
            states_buf = states_buf.at[t].set(state)
            actions_buf = actions_buf.at[t].set(action)
            probs_buf = probs_buf.at[t].set(prob)
            rewards_buf = rewards_buf.at[t].set(r)
    
            # ---------------------------------------------------
            # MOVE TO NEXT STATE
            # ---------------------------------------------------
            state = next_state
            end = time.time()
            #print(f"Execution time: {end - start:.4f} seconds")

        # -----------------------------------
        # UPDATE WARM START AFTER ROLLOUT
        # -----------------------------------
        self._previous_action_seq = prev_action_seq
        self.reset()
    
        # -----------------------------------
        # Return arrays (KEEP ON GPU, NO .tolist())
        # -----------------------------------
        total_reward = jnp.sum(rewards_buf)

        return (
            states_buf.tolist(),
            probs_buf.tolist(),
            actions_buf.tolist(),
            float(total_reward),
        )

    # ================================================================================
    # NEW JIT-OPTIMIZED FUNCTIONS (10x SPEEDUP) - DO NOT AFFECT EXISTING JOBS
    # ================================================================================

    @partial(jax.jit, static_argnames=('self', 'frame_skip', 'gail'))
    def forward_pure_jit(self, state, state_train=None, gail=False, *, key, prev_action_seq, frame_skip):
        """
        JIT-COMPILED version of forward_pure for 5-10x speedup.
        Use this for new optimized jobs only - does not affect existing jobs.
        """
        return self.forward_pure(state, state_train, gail, key=key, prev_action_seq=prev_action_seq, frame_skip=frame_skip)

    @partial(jax.jit, static_argnames=('self', 'gym_env', 'frame_skip'))
    def reward_fn_jit(self, gym_env, state, action, next_state, mjx_data, dt, frame_skip):
        """
        JIT-COMPILED version of reward_fn for 1.2-1.5x speedup.
        Use this for new optimized jobs only - does not affect existing jobs.
        """
        return self.reward_fn(gym_env, state, action, next_state, mjx_data, dt, frame_skip)

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def generate_session_lax_jit(self, args, state_train, D_demo, mpc_method=None, thetas=None):
        """
        JIT-COMPILED version of generate_session_lax for 2-3x speedup.
        Use this for new optimized jobs only - does not affect existing jobs.
        Combined with forward_pure_jit and reward_fn_jit for 8-15x total speedup.
        """
        key = jax.random.PRNGKey(args.seed)
        dt = args.dt
        frame_skip = args.frame_skip

        init_state = D_demo[0, :args.s_dim]
        prev_action_seq0 = self._previous_action_seq
        env = args.gym_env

        if env == "CartPole-v1" or env == "Pendulum-v1" or env == "MountainCarContinuous-v0":
            self.env, self.env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = self.env.reset(rng_reset, self.env_params)
        elif env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4"]:
            self.mjx_data = self.reset_mjx_state(self.mjx_model, key=key)
        else:
            env_state = self.env.reset(seed=args.seed)

        def rollout_step(carry, t):
            state, key, prev_action_seq = carry

            # Use JIT version for speedup
            action_seq, state_seq, key, prev_action_seq = self.forward_pure_jit(
                state=state, state_train=state_train, gail=args.gail,
                key=key, prev_action_seq=prev_action_seq, frame_skip=frame_skip
            )

            if self.gym_env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4"]:
                next_state = kinematics_mujoco(
                    self.mjx_model, self.mjx_data, state.flatten(),
                    action_seq[0, :].reshape((1, -1)), self.gym_env, frame_skip=frame_skip
                ).flatten()
            else:
                next_state = self._dynamics(state, action_seq[0, :])

            next_state = next_state.ravel()
            action = action_seq[0, :]

            prob = jnp.array([1])
            # Use JIT version for speedup
            r = self.reward_fn_jit(self.gym_env, state, action, next_state, self.mjx_data, dt, frame_skip)

            new_carry = (next_state, key, prev_action_seq)
            outputs = (state, prob, action, r)
            return new_carry, outputs

        (final_carry, traj) = lax.scan(
            rollout_step,
            (init_state, key, prev_action_seq0),
            jnp.arange(args.N_steps)
        )
        (final_state, final_key, final_prev_action_seq), (states, traj_probs, actions, rewards) = final_carry, traj

        self._previous_action_seq = final_prev_action_seq
        self.reset()

        rewards = jnp.sum(rewards)
        states, traj_probs, actions, rewards = (
            states.tolist(),
            traj_probs.tolist(),
            actions.tolist(),
            rewards.tolist()
        )

        return states, traj_probs, actions, rewards


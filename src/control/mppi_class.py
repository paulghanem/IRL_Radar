
from __future__ import annotations

from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

from src.control.dynamics import kinematics,kinematics_mujoco
from src.objective_fns.cost_to_go_fns import get_cost
from cost_jax import get_gradients,get_hessian

import os.path as osp

import numpy as np
import math
from tqdm.auto import tqdm
from copy import deepcopy
#from utils.models import load_neural_network
import gymnax
import gymnasium as gym
import pdb
from cost_jax import apply_model, apply_model_AIRL, update_model
from mujoco import mjx 

@jax.jit
def update_theta(theta, P_theta, Q_theta, hessian_d, hessian_s, gradient_d, gradient_s):
    P_theta = jnp.linalg.inv(jnp.linalg.inv(P_theta + Q_theta) + hessian_d - hessian_s)
    #P_theta = jnp.linalg.inv( hessian_d - hessian_s)
    theta = theta - jnp.matmul(P_theta, gradient_d - gradient_s)
    return theta,P_theta
def load_config(config_fname, seed_id=0, lrate=None):
    """Load training configuration and random seed of experiment."""
    import yaml
    import re
    from dotmap import DotMap

    def load_yaml(config_fname: str) -> dict:
        """Load in YAML config file."""
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(config_fname) as file:
            yaml_config = yaml.load(file, Loader=loader)
        return yaml_config

    config = load_yaml(config_fname)
    config["train_config"]["seed_id"] = seed_id
    if lrate is not None:
        if "lr_begin" in config["train_config"].keys():
            config["train_config"]["lr_begin"] = lrate
            config["train_config"]["lr_end"] = lrate
        else:
            try:
                config["train_config"]["opt_params"]["lrate_init"] = lrate
            except Exception:
                pass
    return DotMap(config)

class MPPI:
    """
    Model Predictive Path Integral Control,
    J. Williams et al., T-RO, 2017.
    """

    def __init__(
            self,
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
        self.mjx_model=mjx_model
        self.gym_env=gym_env
        if self.mjx_model is not None :
            self.mjx_data = mjx.make_data(self.mjx_model)

        # noise distribution
        self._covariance = np.zeros((
            self._horizon,
            self._dim_control,
            self._dim_control,
        ))
        self._covariance[:, :, :] = np.diag(sigmas ** 2)
        self._inv_covariance = np.zeros_like(
            self._covariance
        )

        self.covariance = jnp.array(self._covariance)

        for t in range(1, self._horizon):
            self._inv_covariance[t] = np.linalg.inv(self._covariance[t])

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

    # def forward(self,state,state_train=None,gail=False):
    #     for i in range(100):
    #         self.step(state,state_train,gail)
    #
    #     # predivtive state seq
    #     expanded_optimal_action_seq = jnp.tile(self._previous_action_seq,(1, 1, 1))
    #     optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)
    #
    #     return self.step(state,state_train,gail)

    def forward(
            self, state,state_train=None,gail=False
    ):
        """
        Solve the optimal control problem.
        Args:
            state (torch.Tensor): Current state.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of predictive control and state sequence.
        """
        assert state.shape == (self._dim_state,)

        mean_action_seq = self._previous_action_seq.clone()

        # random sampling with reparametrization trick
        self._action_noises = multivariate_normal(self.key, mean=self.zero_mean, cov=self._covariance,
                                                  shape=self._sample_shape)
        self.key,_ = jax.random.split(self.key)

        # noise injection with exploration
        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + self._action_noises[:threshold]
        self._perturbed_action_seqs = jnp.concatenate(
            [inherited_samples, self._action_noises[threshold:]]
        )

        # clamp actions
        self._perturbed_action_seqs = jnp.clip(
            self._perturbed_action_seqs, self._u_min, self._u_max
        )

        # rollout samples in parallel
        # number mppi samples x horizon + 1 x state dim
        #pdb.set_trace()
        
        
        if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d"]:
            if self.gym_env in ["Ant"]:
                state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
            initial_state =  jnp.tile(state,(self._num_samples,1))
            #init_data.replace(qpos=state[:8],qvel=state[8:])
            #batch_mjx_data = jax.tree_util.tree_map(lambda x: jnp.stack([x]*self._num_samples), init_data)
            self._state_seq_batch = jax.vmap(kinematics_mujoco,in_axes=(None,None,0,0,None,None))(self.mjx_model,self.mjx_data,initial_state,self._perturbed_action_seqs,self._dynamics,self.gym_env)
            initial_state=initial_state.reshape((initial_state.shape[0],1,initial_state.shape[1]))
            self._state_seq_batch = jnp.concatenate((initial_state, self._state_seq_batch), axis=1)
            if self.gym_env in ["Ant"]:
                self._state_seq_batch = self._state_seq_batch[:,:,2:]
                state=state[2:]
        else:  
            initial_state =  jnp.tile(state,(self._num_samples, 1,1))
            self._state_seq_batch = jax.vmap(kinematics,in_axes=(0,0,None))(initial_state, self._perturbed_action_seqs,self._dynamics)
        
            self._state_seq_batch = jnp.squeeze(self._state_seq_batch, axis=-2)
            self._state_seq_batch = jnp.concatenate((initial_state, self._state_seq_batch), axis=1)

        # unroll the state seq...
        # for t in range(self._horizon):
        #     self._state_seq_batch[:, t + 1, :] = self._dynamics(
        #         self._state_seq_batch[:, t, :],
        #         self._perturbed_action_seqs[:, t, :],
        #     )

        #compute sample costs
        costs = np.zeros(
            (self._num_samples, self._horizon)
        )

        action_costs = np.zeros(
            (self._num_samples, self._horizon)
        )

        costs=jax.vmap(self._cost_func,in_axes=(1,None))(self._state_seq_batch[:,:-1,:],state_train)
        costs=costs[:,:,0]
        costs=costs.T
     
       


        terminal_costs = self._cost_func(
            self._state_seq_batch[:, -1, :],state_train
        ).ravel()

        # In the original paper, the action cost is added to consider KL div. penalty,
        # but it is easier to tune without it
        costs = (
                jnp.sum(costs, axis=1)
                + terminal_costs
            # + torch.sum(self._lambda * action_costs, axis=1)
        )

        if gail:
            D = jnp.divide(jnp.exp(-costs), (jnp.exp(-costs) + 1))
            costs = -jnp.log(D)


        # calculate weights
        self._weights = jax.nn.softmax(-costs / self._lambda, axis=0)

        # find optimal control by weighted average
        optimal_action_seq = jnp.sum(
            self._weights.reshape(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            axis=0,
        )

        expanded_optimal_action_seq = jnp.tile(self._previous_action_seq,(1, 1, 1))
        optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)
        # update previous actions
        self._previous_action_seq = optimal_action_seq

        return optimal_action_seq, optimal_state_seq


    def _states_prediction(
            self, state, action_seqs
    ):
        
        if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d"]: 
            initial_state=state
            if self.gym_env in ["Ant"]:
                initial_state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
            initial_state=jnp.array(initial_state.reshape((1,-1)),dtype=jnp.float64)
            action_seqs=jnp.array(action_seqs,dtype=jnp.float64)
            state_seqs=jax.vmap(kinematics_mujoco,in_axes=(None,None,0,0,None,None))(self.mjx_model,self.mjx_data,initial_state,action_seqs,self._dynamics,self.gym_env)
            state_seqs= jnp.concatenate((jnp.tile(initial_state,(1,1,1)), state_seqs), axis=1)
            if self.gym_env in ["Ant"]:
                state_seqs=state_seqs[:,:,2:]
        else:
            state_seqs = np.zeros((
                action_seqs.shape[0],
                self._horizon + 1,
                self._dim_state,
            ))
            state_seqs[:, 0, :] = state
        # expanded_optimal_action_seq = action_seq.repeat(1, 1, 1)
       
            for t in range(self._horizon):
                state_seqs[:, t + 1, :] = self._dynamics(
                    state_seqs[:, t, :], action_seqs[:, t, :]
                )
                
        return jnp.array(state_seqs)

    def predict_probs(self, mean, cov, x):
        x = x.T
        mean = mean.T
        pdf = (2 * math.pi) ** (-1) * (jnp.linalg.det(cov)) ** (-1 / 2) * jnp.exp(
            -1 / 2 * jnp.matmul(jnp.matmul((x - mean).T, jnp.linalg.inv(cov)), (x - mean)))
        return pdf

    def generate_session(self, args, state_train, D_demo, mpc_method=None, thetas=None):
        
      
        
        states, traj_probs, actions = [], [], [],

        key = jax.random.PRNGKey(args.seed)
      
        
        np.random.seed(args.seed)
        env=args.gym_env     
        if env=="CartPole-v1" or env=="Pendulum-v1" or env=="MountainCarContinuous-v0":
            env, env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = env.reset(rng_reset, env_params)
        elif env in["HalfCheetah-v4","Ant","Hopper","Walker2d"]:
            self.mjx_data = mjx.make_data(self.mjx_model)
        else:
            env = gym.make(env)
            env_state = env.reset(seed=args.seed)
            
       
        


        state = D_demo[0, :args.s_dim]

       

        #rewards = [true_cost_fn(state)]
        rewards=0

        pbar = tqdm(total=args.N_steps, desc="Starting")

      
        #self.mjx_data = self.mjx_data.replace(qpos=self.mjx_data.qpos.at[0].set(0))
        for step in range(1, args.N_steps + 1):
            states.append(state)
            _, _, rng_step = jax.random.split(key, 3)

            action_seq, state_seq = self.forward(state=state,state_train=state_train, gail=args.gail)

        
          
            
            # env_state.theta=state[0]
            # env_state.theta_dot=state[1]
            # env_state.last_u=state[2]
            # _,_, reward, _, _= env.step(
            #     rng_step, env_state, action_seq[0,:], env_params
            # )
            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d"]: 
                if self.gym_env in ["Ant"]:
                    state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
                state=jnp.array(state,dtype=jnp.float64)
                action_seq=jnp.array(action_seq,dtype=jnp.float64)
                state=kinematics_mujoco(self.mjx_model,self.mjx_data,state.flatten(),action_seq[0,:].reshape((1,-1)),self._dynamics,self.gym_env).flatten()
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
                ctrl_cost = 0.1 * np.sum(np.square(action))
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
                
                
                
            #rewards.append(true_cost_fn(state))
            rewards=rewards+r[0]
            #pdb.set_trace()
            pbar.set_description(
                f"  Total True Reward = {rewards.item():.4f} ,Reward True = {r[0].item():.4f}")
                #f"Reward True = {r[0].item():.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            pbar.update(1)
            # probs_fun = jax.vmap(self.predict_probs, (0, None, 0))
            # prob = self.predict_probs(action_seq[0],
            #                           self.covariance[0],
            #                           ction_seq)

            if args.online:
                state_expert,prob_expert, action_expert = D_demo[step,:args.s_dim], D_demo[step,args.s_dim], D_demo[step,args.s_dim+1:]
    
                if args.airl:
                    grads, loss = apply_model_AIRL(state_train, state, action,state_expert,action_expert,prob,prob_expert,args.UB)
                else:
                    grads, loss = apply_model(state_train, state, action,state_expert,action_expert,prob,prob_expert,args.UB)
    
                state_train = update_model(state_train, grads)
        #pdb.set_trace()
        #rewards = jnp.array(rewards)
        pbar.close()

        self.reset()
   

        return states, traj_probs, actions,rewards

    def RGCL(self, args, params, state_train, D_demo,P_theta, thetas=None):

        # theta = jnp.concatenate((params['Dense_0']['bias'].flatten(), params['Dense_0']['kernel'].flatten(),
        #                          params['Dense_1']['bias'].flatten(), params['Dense_1']['kernel'].flatten()))
        flat_params, treedef = jax.tree_util.tree_flatten(params)
        theta=jnp.concatenate([p.flatten() for p in flat_params])
        #pdb.set_trace()
        n_theta = len(theta)
        P_theta = args.P * jnp.identity(n_theta)

        Q_theta = args.Q* jnp.identity(n_theta)
        states, traj_probs, actions, FIMs = [], [], [], []

        key = jax.random.PRNGKey(args.seed)
        np.random.seed(args.seed)
        env=args.gym_env     
        if env=="CartPole-v1" or env=="Pendulum-v1" or env=="MountainCarContinuous-v0":
            env, env_params = gymnax.make(env)
            _, rng_reset = jax.random.split(key)
            env_state = env.reset(rng_reset, env_params)
        elif env in["HalfCheetah-v4","Ant","Hopper","Walker2d"]:
            self.mjx_data = mjx.make_data(self.mjx_model)
        else:
            env = gym.make(env)
            env_state = env.reset(seed=args.seed)



        state = D_demo[0, :args.s_dim]

        true_cost_fn = get_cost(args.gym_env)

        #rewards = [true_cost_fn(state)]
        rewards=0

        pbar = tqdm(total=args.N_steps, desc="Starting")

        states_expert, actions_expert = D_demo[:, :state.shape[0]], D_demo[:, state.shape[0]:]
        # total_cost = 0
        n_layers=len(params)

        for step in range(1, args.N_steps + 1):
            states.append(state)

            action_seq, state_seq = self.forward(state=state,state_train=state_train, gail=args.gail)
            # step_cost = true_cost_fn(state)
            # total_cost += step_cost
            # pbar.set_description(
            #     f"State = {state} , Action = {action_seq[0].flatten()} , Total True Cost = {total_cost:.4f} , Cost True = {step_cost:.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            # pbar.update(1)

            if self.gym_env in ["HalfCheetah-v4","Ant","Hopper","Walker2d"]: 
               if self.gym_env in ["Ant"]:
                    state=jnp.concat((jnp.reshape(self.mjx_data.qpos[0:2],(2,)),state))
               state=jnp.array(state,dtype=jnp.float64)
               action_seq=jnp.array(action_seq,dtype=jnp.float64)
               state=kinematics_mujoco(self.mjx_model,self.mjx_data,state.flatten(),action_seq[0,:].reshape((1,-1)),self._dynamics,self.gym_env).flatten()
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
                   forward_reward=state[15]
           
            else:
               state = self._dynamics(state, action_seq[0,:])  # , reward, terminated, truncated, info = env.step(action_seq_np[0, :])
           
            state = state.ravel()
            action=action_seq[0,:]
           
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
                ctrl_cost = 0.1 * np.sum(np.square(action))
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
                    break
                    alive_bonus=0
                
                ctrl_cost = 0.001 * np.sum(np.square(action))
                r = forward_reward - ctrl_cost + alive_bonus
                r=r.reshape((1,1))
           
            #rewards.append(true_cost_fn(state))
            #pdb.set_trace()
            rewards=rewards+r[0]
           
            pbar.set_description(
                f"  Total True Reward = {rewards.item():.4f} ,Reward True = {r[0].item():.4f}")
                #f"Reward True = {r[0].item():.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            pbar.update(1)

            gradient_s = get_gradients(state_train, params, state, args.N_steps)
            gradient_d = get_gradients(state_train, params, states_expert[step - 1], args.N_steps)
            hessian_s = get_hessian(state_train, params, state, args.N_steps)
            hessian_d = get_hessian(state_train, params, states_expert[step - 1], args.N_steps)

        
            #pdb.set_trace()
            theta,P_theta= update_theta(theta, P_theta, Q_theta, hessian_d, hessian_s, gradient_d, gradient_s)

            #print(P_theta)
            # params['Dense_0']['bias'] = theta[:len(params['Dense_0']['bias'])]
            # params['Dense_0']['kernel'] = theta[len(params['Dense_0']['bias']):len(params['Dense_0']['bias']) +
            #                                                                    params['Dense_0']['kernel'].shape[0] *
            #                                                                    params['Dense_0']['kernel'].shape[
            #                                                                        1]].reshape(
            #     params['Dense_0']['kernel'].shape)
                                                                                       
            # params['Dense_1']['bias'] = theta[len(params['Dense_0']['bias']) + params['Dense_0']['kernel'].shape[0] *
            #                                   params['Dense_0']['kernel'].shape[1]:len(params['Dense_0']['bias']) +
            #                                                                        params['Dense_0']['kernel'].shape[
            #                                                                            0] *
            #                                                                        params['Dense_0']['kernel'].shape[
            #                                                                            1] + len(
            #     params['Dense_1']['bias'])]
            # params['Dense_1']['kernel'] = theta[len(params['Dense_0']['bias']) + params['Dense_0']['kernel'].shape[0] *
            #                                     params['Dense_0']['kernel'].shape[1] + len(
            #     params['Dense_1']['bias']):].reshape(-1, 1)
            counter=0
            for i in range (n_layers):
                for j in ['bias','kernel']:
                    params['Dense_'+str(i)][j]=theta[counter:counter+len(params['Dense_'+str(i)][j].flatten())].reshape(params['Dense_'+str(i)][j].shape)
                    counter+=len(params['Dense_'+str(i)][j].flatten())
                    
        #rewards = np.array(rewards)
        pbar.close()
        self.reset()
    

        return states, traj_probs, actions,rewards,P_theta


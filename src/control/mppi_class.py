"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

from src.control.dynamics import kinematics
from src.objective_fns.cost_to_go_fns import get_cost
from cost_jax import get_gradients,get_hessian

import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

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

        # compute sample costs
        costs = np.zeros(
            (self._num_samples, self._horizon)
        )

        action_costs = np.zeros(
            (self._num_samples, self._horizon)
        )

        for t in range(self._horizon):


            costs[:, t] = self._cost_func(
                self._state_seq_batch[:, t, :],state_train
            ).ravel()
            action_costs[:, t] = (
                    mean_action_seq[t]
                    @ self._inv_covariance[t]
                    @ self._perturbed_action_seqs[:, t].T
            )


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


        state = D_demo[0, :args.s_dim]

        true_cost_fn = get_cost(args.gym_env)

        rewards = [true_cost_fn(state)]

        pbar = tqdm(total=args.N_steps, desc="Starting")

        total_cost = 0

        for step in range(1, args.N_steps + 1):
            states.append(state)

            action_seq, state_seq = self.forward(state=state,state_train=state_train, gail=args.gail)

            step_cost = true_cost_fn(state)
            total_cost += step_cost
            pbar.set_description(
                f"State = {state} , Action = {action_seq[0].flatten()} , Total True Cost = {total_cost:.4f} , Cost True = {step_cost:.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            pbar.update(1)

            state = self._dynamics(state, action_seq[0,:])  # , reward, terminated, truncated, info = env.step(action_seq_np[0, :])
            state = state.ravel()

            rewards.append(true_cost_fn(state))

            # probs_fun = jax.vmap(self.predict_probs, (0, None, 0))
            # prob = self.predict_probs(action_seq[0],
            #                           self.covariance[0],
            #                           ction_seq)

            prob = jnp.array([1])

            traj_probs.append(prob.flatten())
            actions.append(action_seq[0].flatten())


        # rewards = np.array(rewards)
        pbar.close()

        self.reset()

        return states, traj_probs, actions,rewards

    def RGCL(self, args, params, state_train, D_demo, thetas=None):

        theta = jnp.concatenate((params['Dense_0']['bias'].flatten(), params['Dense_0']['kernel'].flatten(),
                                 params['Dense_1']['bias'].flatten(), params['Dense_1']['kernel'].flatten()))
        n_theta = len(theta)
        P_theta = 1e-1 * jnp.identity(n_theta)

        Q_theta = 1e-4 * jnp.identity(n_theta)
        states, traj_probs, actions, FIMs = [], [], [], []

        key = jax.random.PRNGKey(args.seed)
        np.random.seed(args.seed)




        state = D_demo[0, :args.s_dim]

        true_cost_fn = get_cost(args.gym_env)

        rewards = [true_cost_fn(state)]

        pbar = tqdm(total=args.N_steps, desc="Starting")

        states_expert, actions_expert = D_demo[:, :state.shape[0]], D_demo[:, state.shape[0]:]
        total_cost = 0

        for step in range(1, args.N_steps + 1):
            states.append(state)

            action_seq, state_seq = self.forward(state=state,state_train=state_train, gail=args.gail)
            step_cost = true_cost_fn(state)
            total_cost += step_cost
            pbar.set_description(
                f"State = {state} , Action = {action_seq[0].flatten()} , Total True Cost = {total_cost:.4f} , Cost True = {step_cost:.4f} ,  Cost Estimated = {state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel().item():.4f}")
            pbar.update(1)

            state = self._dynamics(state, action_seq[0,:])  # , reward, terminated, truncated, info = env.step(action_seq_np[0, :])
            state = state.ravel()

            gradient_s = get_gradients(state_train, params, state, args.N_steps)
            gradient_d = get_gradients(state_train, params, states_expert[step - 1], args.N_steps)
            hessian_s = get_hessian(state_train, params, state, args.N_steps)
            hessian_d = get_hessian(state_train, params, states_expert[step - 1], args.N_steps)

            P_theta = jnp.linalg.inv(jnp.linalg.inv(P_theta + Q_theta) + hessian_d - hessian_s)
            # print(gradient_d)
            # print(gradient_s)

            theta = theta - jnp.matmul(P_theta, gradient_d - gradient_s)

            params['Dense_0']['bias'] = theta[:len(params['Dense_0']['bias'])]
            params['Dense_0']['kernel'] = theta[len(params['Dense_0']['bias']):len(params['Dense_0']['bias']) +
                                                                               params['Dense_0']['kernel'].shape[0] *
                                                                               params['Dense_0']['kernel'].shape[
                                                                                   1]].reshape(
                params['Dense_0']['kernel'].shape)
            params['Dense_1']['bias'] = theta[len(params['Dense_0']['bias']) + params['Dense_0']['kernel'].shape[0] *
                                              params['Dense_0']['kernel'].shape[1]:len(params['Dense_0']['bias']) +
                                                                                   params['Dense_0']['kernel'].shape[
                                                                                       0] *
                                                                                   params['Dense_0']['kernel'].shape[
                                                                                       1] + len(
                params['Dense_1']['bias'])]
            params['Dense_1']['kernel'] = theta[len(params['Dense_0']['bias']) + params['Dense_0']['kernel'].shape[0] *
                                                params['Dense_0']['kernel'].shape[1] + len(
                params['Dense_1']['bias']):].reshape(-1, 1)

        #rewards = np.array(rewards)
        pbar.close()
        self.reset()

        return states, traj_probs, actions,total_cost


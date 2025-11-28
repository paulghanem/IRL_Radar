"""
Unified PPO Implementation for IRL with GCL/AIRL/GAIL Integration

This module provides a PPO agent that works across all environments
and can replace MPPI in the main IRL training loop.

Key Features:
- Supports all environments (CartPole, Pendulum, MountainCar, Walker2d, MuJoCo envs)
- Same interface as MPPI (generate_session_lax method)
- Integrates with cost function learning (GCL, AIRL, GAIL, SQIL)
- Can use learned cost function as reward signal or environment reward
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from flax.training import train_state
import optax
import math
from typing import Tuple, Any, Optional
from functools import partial

from src.control.buffer import RolloutBuffer
from src.control.dynamics import kinematics, kinematics_mujoco
from mujoco import mjx
import pdb


class ActorNetwork(nn.Module):
    """Actor network for continuous control"""
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (1, self.action_dim))
        return mu, log_std


class CriticNetwork(nn.Module):
    """Critic network (value function)"""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value


class UnifiedPPO:
    """
    Unified PPO implementation that can replace MPPI in main.py

    Supports:
    - All gym environments (CartPole, Pendulum, MountainCar, Walker2d, HalfCheetah, etc.)
    - Integration with learned cost functions from IRL
    - Same interface as MPPI for drop-in replacement
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        args,
        state_train=None,  # Cost function training state (for IRL integration)
        dynamics=None,      # Dynamics function for simple environments
        mjx_model=None,     # MuJoCo model for MuJoCo environments
        gym_env: str = "CartPole-v1",
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        rollout_length: int = 2048,
        buffer_mix: int = 20,
        use_learned_cost: bool = False  # Use learned cost as reward
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.state_train = state_train
        self._dynamics = dynamics
        self.mjx_model = mjx_model
        self.gym_env = gym_env
        self.rollout_length = rollout_length
        self.hidden_dim = hidden_dim
        self.use_learned_cost = use_learned_cost

        # Initialize MuJoCo data if using MuJoCo
        if self.mjx_model is not None:
            self.mjx_data = mjx.make_data(self.mjx_model)

        # Initialize networks
        self.actor_net = ActorNetwork(action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic_net = CriticNetwork(hidden_dim=hidden_dim)

        # Initialize parameters
        key = jax.random.PRNGKey(args.seed if hasattr(args, 'seed') else 42)
        key_actor, key_critic = jax.random.split(key)

        dummy_state = jnp.zeros((1, state_dim))
        actor_params = self.actor_net.init(key_actor, dummy_state)['params']
        critic_params = self.critic_net.init(key_critic, dummy_state)['params']

        # Create train states
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_net.apply,
            params=actor_params,
            tx=optax.adam(lr_actor)
        )

        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic_net.apply,
            params=critic_params,
            tx=optax.adam(lr_critic)
        )

        # Rollout buffer
        self.buffer = RolloutBuffer.create(
            buffer_size=rollout_length,
            state_shape=(state_dim,),
            action_shape=(action_dim,),
            mix=buffer_mix
        )

    def calculate_log_pi(self, log_std, noise, action):
        """Calculate log probability of action under Gaussian policy with tanh squashing"""
        gaussian_log_prob = jnp.sum(
            -0.5 * jnp.power(noise, 2) - log_std, axis=-1
        ) - 0.5 * math.log(2 * math.pi) * log_std.shape[-1]

        # Correction for tanh squashing
        return gaussian_log_prob - jnp.sum(
            jnp.log(1 - jnp.power(action, 2) + 1e-6), axis=-1
        )

    def atanh(self, x):
        """Inverse tanh"""
        return 0.5 * (jnp.log(1 + x + 1e-6) - jnp.log(1 - x + 1e-6))

    def evaluate_log_pi(self, states, actions, params):
        """Evaluate log probability for given states and actions"""
        mu, log_std = self.actor_net.apply({'params': params}, states)
        noise = (self.atanh(actions) - mu) / (jnp.exp(log_std) + 1e-8)
        return self.calculate_log_pi(log_std, noise, actions)

    def sample_action(self, state, params, key):
        """Sample action from policy"""
        mu, log_std = self.actor_net.apply({'params': params}, state[None, :])
        noise = jax.random.normal(key, mu.shape)
        u = mu + jnp.exp(log_std) * noise
        action = jnp.tanh(u)
        log_pi = self.calculate_log_pi(log_std, noise, action)
        return action.flatten(), log_pi.flatten()

    def reset_mjx_state(self, mjx_model, key=None, noise_scale=0.01):
        """
        Recreate a fresh mjx.Data object from a given mjx.Model.
        Optionally add small random noise to qpos/qvel for exploration.
        """
        data = mjx.make_data(mjx_model)
        qpos0 = jnp.zeros_like(data.qpos)
        qvel0 = jnp.zeros_like(data.qvel)

        if key is not None:
            key_qpos, key_qvel = jax.random.split(key)
            qpos0 = qpos0 + noise_scale * jax.random.normal(key_qpos, shape=qpos0.shape)
            qvel0 = qvel0 + noise_scale * jax.random.normal(key_qvel, shape=qvel0.shape)

        data = data.replace(qpos=qpos0, qvel=qvel0)
        mjx.forward(mjx_model, data)
        return data

    def compute_reward(self, state, action, next_state, forward_reward=None):
        """
        Compute true environment reward for PPO training

        Always uses the actual environment reward, not the learned cost function.
        This ensures PPO learns from true rewards while the IRL cost function
        is used only for trajectory evaluation in the outer loop.
        """
        # Use true environment reward
        if self.gym_env == "CartPole-v1":
            x = next_state[0]
            theta = next_state[2]
            x_threshold = 2.4
            theta_threshold = 12 * 2 * math.pi / 360

            terminated = (
                (x < -x_threshold) | (x > x_threshold) |
                (theta < -theta_threshold) | (theta > theta_threshold)
            )
            r = jnp.where(terminated, 0.0, 1.0)

        elif self.gym_env == "Pendulum-v1":
            x = next_state[0]
            y = next_state[1]
            theta = jnp.atan2(y, x)
            theta_dot = next_state[2]
            r = -(jnp.pow(theta, 2) + 0.1 * jnp.pow(theta_dot, 2) + 0.001 * jnp.pow(action, 2))

        elif self.gym_env == "MountainCarContinuous-v0":
            r = -0.1 * jnp.pow(action, 2).sum()
            goal_position = 0.45
            x = next_state[0]
            r = r + jnp.where(x >= goal_position, 100.0, 0.0)

        elif self.gym_env == "HalfCheetah-v4":
            ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost

        elif self.gym_env == "Ant":
            alive_bonus = 1.0
            alive_bonus = jnp.where((next_state[1] < 0.2) | (next_state[1] > 1), 0.0, alive_bonus)
            ctrl_cost = 0.5 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost + alive_bonus

        elif self.gym_env == "Hopper":
            alive_bonus = 1.0
            # Check state limits
            bad_state = jnp.any(next_state[2:] < -100) | jnp.any(next_state[2:] > 100)
            bad_angle = (next_state[2] < -0.2) | (next_state[2] > 0.2)
            bad_height = next_state[1] < 0.7
            alive_bonus = jnp.where(bad_state | bad_angle | bad_height, 0.0, alive_bonus)
            ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost + alive_bonus

        elif self.gym_env == "Walker2d":
            alive_bonus = 1.0
            fall_cond = (
                (jnp.abs(next_state[2]) > 1.0) |
                (next_state[1] < 0.8) |
                (next_state[1] > 2.0)
            )
            alive_bonus = jnp.where(fall_cond, 0.0, alive_bonus)
            ctrl_cost = 0.001 * jnp.sum(jnp.square(action))
            r = forward_reward - ctrl_cost + alive_bonus

        elif self.gym_env == "Humanoid-v4":
            alive_bonus = 5.0
            alive_bonus = jnp.where((next_state[2] < 1) | (next_state[2] > 2), 0.0, alive_bonus)
            quad_impact_cost = 0.5e-6 * jnp.square(self.mjx_data.cfrc_ext).sum()
            quad_impact_cost = jnp.minimum(quad_impact_cost, 10.0)
            ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
            r = 1.25 * forward_reward - ctrl_cost - quad_impact_cost + alive_bonus

        else:
            r = 0.0

        return r

    def generate_session_lax(self, args, state_train, D_demo, iteration=0):
        """
        Generate trajectory using PPO policy

        This method has the same signature as MPPI.generate_session_lax()
        to enable drop-in replacement.

        Args:
            args: Configuration arguments
            state_train: Cost function training state (updated during training)
            D_demo: Expert demonstrations
            iteration: Current iteration number (for varying random seed)

        Returns:
            states, probs, actions, total_reward (compatible with MPPI interface)
        """
        # Update cost function training state if provided
        if state_train is not None:
            self.state_train = state_train

        # Use different seed for each iteration to get diversity
        key = jax.random.PRNGKey(args.seed + iteration if hasattr(args, 'seed') else 42 + iteration)
        init_state = D_demo[0, :args.s_dim]

        # Reset MuJoCo state if using MuJoCo
        if self.mjx_model is not None:
            reset_data = self.reset_mjx_state(self.mjx_model, key=key)

        # Get frame skip and dt from args
        frame_skip = args.frame_skip if hasattr(args, 'frame_skip') else 1
        dt = args.dt if hasattr(args, 'dt') else 0.01

        def rollout_step(carry, t):
            state, key, buffer, mjx_data = carry

            # Split RNG
            key, subkey = jax.random.split(key)

            # Sample action from policy
            action, log_pi = self.sample_action(state, self.actor_state.params, subkey)

            # Environment step
            if self.gym_env in ["HalfCheetah-v4", "Ant", "Hopper", "Walker2d", "Humanoid-v4", "Swimmer"]:
                # MuJoCo environments
                next_state = kinematics_mujoco(
                    self.mjx_model, mjx_data, state, action.reshape((1, -1)),
                    self.gym_env, frame_skip=frame_skip
                ).flatten()

                # Compute forward reward based on environment
                if self.gym_env == "HalfCheetah-v4":
                    forward_reward = (next_state[0] - state[0]) / (frame_skip * dt)
                elif self.gym_env == "Hopper":
                    forward_reward = next_state[6]
                elif self.gym_env == "Walker2d":
                    forward_reward = next_state[9]
                elif self.gym_env == "Ant":
                    forward_reward = next_state[13]
                elif self.gym_env == "Humanoid-v4":
                    forward_reward = next_state[0]  # Placeholder
                else:
                    forward_reward = 0.0

            else:
                # Simple environments with analytical dynamics
                next_state = self._dynamics(state, action)
                next_state = jnp.atleast_1d(next_state).flatten()
                forward_reward = None

            # Compute reward
            reward = self.compute_reward(state, action, next_state, forward_reward)
            reward_array = jnp.atleast_1d(reward)[:1]

            # Check termination
            done = jnp.array([0.0])

            # Check environment-specific termination conditions
            if self.gym_env == "Walker2d":
                fall_cond = (
                    (jnp.abs(next_state[2]) > 1.0) |
                    (next_state[1] < 0.8) |
                    (next_state[1] > 2.0)
                )
                done = jnp.array([jnp.where(fall_cond, 1.0, 0.0)])
            elif self.gym_env == "CartPole-v1":
                x = next_state[0]
                theta = next_state[2]
                x_threshold = 2.4
                theta_threshold = 12 * 2 * math.pi / 360
                terminated = (
                    (x < -x_threshold) | (x > x_threshold) |
                    (theta < -theta_threshold) | (theta > theta_threshold)
                )
                done = jnp.array([jnp.where(terminated, 1.0, 0.0)])

            # Store transition in buffer
            buffer = buffer.append(
                state, action, reward_array,
                done, log_pi, next_state
            )

            # Reset state if done (for MuJoCo environments)
            if self.mjx_model is not None:
                mjx_data_new = jax.tree_util.tree_map(
                    lambda x, y: jnp.where(done[0] > 0.5, x, y), reset_data, mjx_data
                )
                next_state = jax.tree_util.tree_map(
                    lambda x, y: jnp.where(done[0] > 0.5, x, y),
                    jnp.concatenate([reset_data.qpos, reset_data.qvel]), next_state
                )
            else:
                mjx_data_new = mjx_data
                # For simple environments, DON'T reset - let it continue from failed state
                # This allows proper reward accumulation and policy comparison
                # (Resetting artificially inflates rewards for bad policies)
                # next_state = jnp.where(done[0] > 0.5, init_state, next_state)
                # Keep next_state as is - don't reset

            carry = (next_state, key, buffer, mjx_data_new)
            # Output format: (state, prob, action, reward)
            # prob is set to 1.0 for compatibility with preprocess_traj
            outputs = (state, jnp.array([1.0]), action, reward, log_pi, done)
            return carry, outputs

        # Run rollout
        init_mjx_data = self.mjx_data if self.mjx_model is not None else None
        init_carry = (init_state, key, self.buffer, init_mjx_data)

        final_carry, traj = lax.scan(
            rollout_step,
            init_carry,
            jnp.arange(args.N_steps)
        )

        # Update buffer and mjx_data
        _, _, final_buffer, final_mjx_data = final_carry
        self.buffer = final_buffer
        if self.mjx_model is not None:
            self.mjx_data = final_mjx_data

        # Extract trajectory components
        states, probs, actions, rewards, log_pis, dones = traj

        # Compute total reward
        total_reward = jnp.sum(rewards)

        # Return format compatible with MPPI: (states, probs, actions, total_reward)
        return (
            states.tolist(),
            probs.tolist(),
            actions.tolist(),
            total_reward.tolist()
        )

    def update_ppo(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        log_probs_old: jnp.ndarray,
        next_states: jnp.ndarray,
        gamma: float = 0.99,
        lam: float = 0.97,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        num_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5
    ):
        """
        Update PPO policy and value function using collected experience

        Args:
            states: State observations
            actions: Actions taken
            rewards: Rewards received
            dones: Done flags
            log_probs_old: Log probabilities under old policy
            next_states: Next state observations
            gamma: Discount factor
            lam: GAE lambda
            clip_eps: PPO clipping epsilon
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            num_epochs: Number of update epochs
            batch_size: Minibatch size
            max_grad_norm: Maximum gradient norm for clipping
        """
        # Ensure all arrays have correct shapes
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards).flatten()
        dones = jnp.array(dones).flatten()
        log_probs_old = jnp.array(log_probs_old).flatten()
        next_states = jnp.array(next_states)

        n_samples = states.shape[0]
        if n_samples < batch_size:
            print(f"Warning: Not enough samples ({n_samples}) for batch size ({batch_size}). Skipping update.")
            return

        # Compute values
        values = self.critic_state.apply_fn(
            {'params': self.critic_state.params}, states
        ).flatten()
        next_values = self.critic_state.apply_fn(
            {'params': self.critic_state.params}, next_states
        ).flatten()

        # Compute GAE advantages
        def compute_gae(rewards, values, next_values, dones):
            deltas = rewards + gamma * (1.0 - dones) * next_values - values
            advantages = jnp.zeros_like(deltas)
            gae = 0.0
            for i in range(len(deltas) - 1, -1, -1):
                gae = deltas[i] + gamma * lam * (1.0 - dones[i]) * gae
                advantages = advantages.at[i].set(gae)
            return advantages

        advantages = compute_gae(rewards, values, next_values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update for multiple epochs
        for epoch in range(num_epochs):
            # Shuffle indices
            key = jax.random.PRNGKey(epoch)
            indices = jax.random.permutation(key, n_samples)

            # Create minibatches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                mb_indices = indices[start:end]

                # Actor loss
                def actor_loss_fn(params):
                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_logp = log_probs_old[mb_indices]
                    mb_adv = advantages[mb_indices]

                    # Compute new log probs
                    new_logp = self.evaluate_log_pi(mb_states, mb_actions, params)

                    # PPO clipped loss
                    ratio = jnp.exp(new_logp - mb_old_logp)
                    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                    loss_pi = -jnp.mean(jnp.minimum(ratio * mb_adv, clipped * mb_adv))

                    # Entropy bonus
                    mu, log_std = self.actor_net.apply({'params': params}, mb_states)
                    entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))

                    return loss_pi - ent_coef * entropy

                # Critic loss
                def critic_loss_fn(params):
                    mb_states = states[mb_indices]
                    mb_returns = returns[mb_indices]

                    v = self.critic_state.apply_fn({'params': params}, mb_states).flatten()
                    vf_loss = jnp.mean((mb_returns - v) ** 2)
                    return vf_coef * vf_loss

                # Update actor
                actor_grads = jax.grad(actor_loss_fn)(self.actor_state.params)
                # Gradient clipping
                actor_grads = jax.tree_util.tree_map(
                    lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm),
                    actor_grads
                )
                self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)

                # Update critic
                critic_grads = jax.grad(critic_loss_fn)(self.critic_state.params)
                # Gradient clipping
                critic_grads = jax.tree_util.tree_map(
                    lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm),
                    critic_grads
                )
                self.critic_state = self.critic_state.apply_gradients(grads=critic_grads)

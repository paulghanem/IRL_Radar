"""
PPO implementation for SimplifiedWalker2d with GCL integration

This module provides a PPO agent that works with SimplifiedWalker2d dynamics
and integrates seamlessly with GCL (Guided Cost Learning).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from flax.training import train_state
import optax
import math
from typing import Tuple, Any
from functools import partial

from src.control.buffer import RolloutBuffer
from src.control.simplified_walker import SimplifiedWalker


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


class PPOWalker:
    """PPO agent for SimplifiedWalker2d"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        walker: SimplifiedWalker,
        args,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        rollout_length: int = 2048,
        buffer_mix: int = 20
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.walker = walker
        self.args = args
        self.rollout_length = rollout_length
        self.hidden_dim = hidden_dim

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

    @partial(jax.jit, static_argnums=(0,))
    def compute_walker_reward(self, state, action, next_state):
        """
        Compute Walker2d-style reward using SimplifiedWalker2d

        Reward components:
        - Forward velocity (encourage forward motion)
        - Alive bonus (encourage staying upright)
        - Control cost (discourage large actions)
        """
        # Forward velocity (x-velocity is at index 9 in Walker2d state)
        forward_velocity = next_state[9]
        forward_reward = forward_velocity

        # Alive bonus (check if walker is still standing)
        z_height = next_state[1]  # Height (index 1)
        angle = next_state[2]     # Root angle (index 2)

        alive = (z_height > 0.8) & (z_height < 2.0) & (jnp.abs(angle) < 1.0)
        alive_bonus = jnp.where(alive, 1.0, 0.0)

        # Control cost
        ctrl_cost = 0.001 * jnp.sum(jnp.square(action))

        # Total reward
        reward = forward_reward + alive_bonus - ctrl_cost

        return reward

    def generate_session_lax(self, args, state_train, D_demo, iteration=0):
        """
        Generate trajectory using PPO policy with SimplifiedWalker2d dynamics

        Args:
            args: Configuration arguments
            state_train: Cost function training state (for GCL compatibility)
            D_demo: Expert demonstrations
            iteration: Current iteration number (for varying random seed)

        Returns:
            states, probs, actions, total_reward
        """
        # Use different seed for each iteration to get diversity
        key = jax.random.PRNGKey(args.seed + iteration)
        init_state = D_demo[0, :args.s_dim]

        def rollout_step(carry, t):
            state, key, buffer = carry

            # Split RNG
            key, subkey = jax.random.split(key)

            # Sample action from policy
            action, log_pi = self.sample_action(state, self.actor_state.params, subkey)

            # Environment step using SimplifiedWalker2d
            next_state = self.walker.step(state, action)

            # Compute reward
            reward = self.compute_walker_reward(state, action, next_state)

            # Check termination
            z_height = next_state[1]
            angle = next_state[2]
            done = (z_height < 0.8) | (z_height > 2.0) | (jnp.abs(angle) > 1.0)
            done_array = jnp.array([done], dtype=jnp.float32)

            # Store transition in buffer
            buffer = buffer.append(
                state, action, jnp.array([reward]),
                done_array, log_pi, next_state
            )

            # Reset state if done
            next_state = jnp.where(done, init_state, next_state)

            carry = (next_state, key, buffer)
            outputs = (state, jnp.array([1.0]), action, reward, log_pi, done_array)
            return carry, outputs

        # Run rollout
        init_carry = (init_state, key, self.buffer)

        final_carry, traj = lax.scan(
            rollout_step,
            init_carry,
            jnp.arange(args.N_steps)
        )

        # Update buffer
        _, _, final_buffer = final_carry
        self.buffer = final_buffer

        # Extract trajectory components
        states, probs, actions, rewards, log_pis, dones = traj

        # Convert to lists for compatibility
        total_reward = jnp.sum(rewards)

        return (
            states.tolist(),
            probs.tolist(),
            actions.tolist(),
            total_reward.tolist()
        )

    def update_ppo(
        self,
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
            gamma: Discount factor
            lam: GAE lambda
            clip_eps: PPO clipping epsilon
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            num_epochs: Number of update epochs
            batch_size: Minibatch size
            max_grad_norm: Maximum gradient norm for clipping
        """
        # Get data from buffer
        states, actions, rewards, dones, log_probs_old, next_states = self.buffer.get()

        n_samples = states.shape[0]
        if n_samples < batch_size:
            return  # Not enough data yet

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

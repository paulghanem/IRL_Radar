"""
Simplified PPO implementation for simple environments (CartPole, Pendulum, MountainCar)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from flax.training import train_state
import math

from src.control.buffer import RolloutBuffer
from cost_jax import get_gradients, get_hessian


class PolicyModel(nn.Module):
    """Simple policy network for continuous actions"""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (1, self.action_dim))
        return mu, log_std


class CriticModel(nn.Module):
    """Simple value network"""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value


class SimplePPO:
    """PPO implementation for simple gymnasium environments"""

    def __init__(self, state_dim, action_dim, dynamics, policy_model, policy_net,
                 args, rollout_length=200, value_fn=None):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.rollout_length = rollout_length
        self.args = args
        self.gym_env = args.gym_env
        self._dynamics = dynamics
        self.policy_model = policy_model
        self.policy_net = policy_net
        self.value_fn = value_fn

        # Rollout buffer
        self.buffer = RolloutBuffer.create(
            buffer_size=rollout_length,
            state_shape=(self.state_dim,),
            action_shape=(self.action_dim,),
            mix=20
        )

    def calculate_log_pi(self, log_stds, noises, actions):
        """Calculate log probability of actions"""
        gaussian_log_probs = jnp.sum(
            -0.5 * jnp.power(noises, 2) - log_stds, axis=-1
        ) - 0.5 * math.log(2 * math.pi) * log_stds.shape[-1]

        return gaussian_log_probs - jnp.sum(
            jnp.log(1 - jnp.power(actions, 2) + 1e-6), axis=-1
        )

    def atanh(self, x):
        """Inverse tanh"""
        return 0.5 * (jnp.log(1 + x + 1e-6) - jnp.log(1 - x + 1e-6))

    def evaluate_log_pi(self, states, actions, params):
        """Evaluate log probability for given states and actions"""
        mu, log_std = self.policy_net.apply({'params': params}, states)
        noises = (self.atanh(actions) - mu) / (jnp.exp(log_std) + 1e-8)
        return self.calculate_log_pi(log_std, noises, actions)

    def reward_fn(self, state, action, next_state):
        """Compute reward for the environment - always returns a scalar JAX array"""
        if self.gym_env == "CartPole-v1":
            x = next_state[..., 0] if next_state.ndim > 0 else next_state[0]
            theta = next_state[..., 2] if next_state.ndim > 0 else next_state[2]
            x_threshold = 2.4
            theta_threshold = 12 * 2 * math.pi / 360

            terminated = (
                (x < -x_threshold) | (x > x_threshold) |
                (theta < -theta_threshold) | (theta > theta_threshold)
            )
            r = jnp.where(terminated, 0.0, 1.0)
            # Make sure it's a 0-d or 1-d array
            r = jnp.squeeze(jnp.atleast_1d(r))

        elif self.gym_env == "Pendulum-v1":
            x = next_state[0]
            y = next_state[1]
            theta = jnp.atan2(y, x)
            theta_dot = next_state[2]
            r = -(jnp.pow(theta, 2) + 0.1 * jnp.pow(theta_dot, 2) + 0.001 * jnp.pow(action, 2))

        elif self.gym_env == "MountainCarContinuous-v0":
            r = -0.1 * jnp.pow(action, 2)
            goal_position = 0.45
            x = next_state[0]
            r = r + jnp.where(x >= goal_position, 100.0, 0.0)

        else:
            r = 0.0

        return r

    def generate_session_lax(self, args, D_demo, frame_skip=1, dt=0.02, **kwargs):
        """Generate trajectory using PPO policy with lax.scan"""

        key = jax.random.PRNGKey(args.seed)
        init_state = D_demo[0, :args.s_dim]

        def rollout_step(carry, t):
            state, key, buffer = carry

            # Split RNG
            key, subkey = jax.random.split(key)

            # Sample action from policy
            mu, log_std = self.policy_net.apply(
                {'params': self.policy_model.params},
                state[None, :]
            )
            noise = jax.random.normal(subkey, mu.shape)
            u = mu + jnp.exp(log_std) * noise
            action = jnp.tanh(u)
            logp = self.calculate_log_pi(log_std, noise, action)

            # Environment step
            next_state_raw = self._dynamics(state, action.flatten())
            # Ensure next_state is 1D (state_dim,)
            next_state = jnp.atleast_1d(next_state_raw).flatten()

            # Compute reward
            r = self.reward_fn(state, action.flatten(), next_state)

            # Ensure reward is (1,) shape for buffer
            r_array = jnp.atleast_1d(r)[:1]  # Take first element and make shape (1,)

            # Check done
            done_array = jnp.array([0.0])  # False as 0.0

            # Store transition
            buffer = buffer.append(
                state, action.flatten(), r_array,
                done_array, logp.flatten(), next_state
            )

            # Update carry
            carry = (next_state, key, buffer)
            outputs = (state, next_state, action[0], r, logp, done_array)
            return carry, outputs

        # Run rollout
        init_carry = (init_state, key, self.buffer)

        final_carry, traj = lax.scan(
            rollout_step,
            init_carry,
            jnp.arange(self.rollout_length)
        )

        # Update buffer
        final_state, _, final_buffer = final_carry
        self.buffer = final_buffer

        return

    def update_ppo(self, states, actions, rewards, dones, log_probs_old, next_states,
                   gamma=0.99, lam=0.97, clip_eps=0.2, vf_coef=1.0, ent_coef=0.0,
                   num_epochs=10, batch_size=64):
        """Update PPO policy and value function"""

        n_samples = states.shape[0]

        def get_advantages(rewards, values, next_values, dones):
            """Compute GAE advantages"""
            deltas = rewards + gamma * (1.0 - dones) * next_values - values

            adv = []
            gae = 0.0
            for delta, done in zip(deltas[::-1], dones[::-1]):
                gae = delta + gamma * lam * (1.0 - done) * gae
                adv.insert(0, gae)
            return jnp.array(adv)

        # Compute values
        if self.value_fn:
            values = self.value_fn.apply_fn({'params': self.value_fn.params}, states).flatten()
            next_values = self.value_fn.apply_fn({'params': self.value_fn.params}, next_states).flatten()
        else:
            values = jnp.zeros_like(rewards)
            next_values = jnp.zeros_like(rewards)

        # Compute advantages
        advantages = get_advantages(rewards, values, next_values, dones)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_loss_fn(params, minibatch_idxs):
            """Actor loss with PPO clipping"""
            s = states[minibatch_idxs]
            a = actions[minibatch_idxs]
            old_logp = jnp.array(log_probs_old)[minibatch_idxs]
            adv = advantages[minibatch_idxs]

            mu, log_std = self.policy_net.apply({'params': params}, s)
            logp = self.evaluate_log_pi(s, a, params)

            ratio = jnp.exp(logp - old_logp)
            clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            loss_pi = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))

            # Entropy bonus
            entropy_per_sample = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
            entropy = jnp.mean(entropy_per_sample)

            loss = loss_pi - ent_coef * entropy
            return loss

        def critic_loss_fn(params, minibatch_idxs):
            """Value function loss"""
            s = states[minibatch_idxs]
            ret = returns[minibatch_idxs]

            v = self.value_fn.apply_fn({'params': params}, s).squeeze()
            vf_loss = jnp.mean((ret - v) ** 2)
            loss = vf_coef * vf_loss
            return loss

        # Update for multiple epochs
        mb = jnp.arange(n_samples)
        for _ in range(num_epochs):
            # Update actor
            actor_grads = jax.grad(actor_loss_fn)(self.policy_model.params, mb)
            self.policy_model = self.policy_model.apply_gradients(grads=actor_grads)

            # Update critic
            if self.value_fn:
                critic_grads = jax.grad(critic_loss_fn)(self.value_fn.params, mb)
                self.value_fn = self.value_fn.apply_gradients(grads=critic_grads)

    def RGCL_lax(self, args, params, state_train, initial_state, D_demo, P_theta_in, thetas=None,
                 ppo_update_freq=10, ppo_batch_size=32, gamma=0.99, lam=0.97, clip_eps=0.2):
        """
        RGCL (Recursive Guided Cost Learning) with PPO policy updates.

        Performs Kalman-style recursive updates on the cost function parameters
        using gradients and Hessians, while updating PPO policy with a moving window.

        Args:
            args: Configuration arguments
            params: Initial cost function parameters
            state_train: Cost function training state
            initial_state: Initial environment state
            D_demo: Expert demonstration data
            P_theta_in: Initial covariance matrix for parameters
            thetas: Optional theta parameters (unused)
            ppo_update_freq: Update PPO every N steps (default: 10)
            ppo_batch_size: Batch size for PPO updates (default: 32)
            gamma: Discount factor for PPO (default: 0.99)
            lam: GAE lambda for PPO (default: 0.97)
            clip_eps: PPO clip epsilon (default: 0.2)

        Returns:
            states: List of visited states
            traj_probs: List of action log probabilities
            actions: List of executed actions
            rewards: Total accumulated reward
            P_theta: Final covariance matrix
            params: Updated cost function parameters
        """

        # Flatten cost function parameters into theta vector
        flat_params, treedef = jax.tree_util.tree_flatten(params)
        theta0 = jnp.concatenate([p.reshape(-1) for p in flat_params])
        n_theta = theta0.size
        key = jax.random.PRNGKey(args.seed)

        # Initialize covariance matrices
        if args.diagonal:
            raise ValueError("Diagonal RGCL not implemented. Use full matrix version.")
        else:
            P0 = args.P * jnp.eye(n_theta)
            Q = args.Q * jnp.eye(n_theta)

        # Extract expert data
        expert_states = D_demo[:, :args.s_dim]
        expert_actions = D_demo[:, args.s_dim:args.s_dim + args.a_dim]

        # Initial state
        init_state = initial_state
        dt = args.dt if hasattr(args, 'dt') else 0.02
        frame_skip = args.frame_skip if hasattr(args, 'frame_skip') else 1

        # Initialize experience buffer for PPO updates
        from src.control.buffer import RolloutBuffer
        buffer_size = max(ppo_batch_size, ppo_update_freq * 2)
        experience_buffer = RolloutBuffer.create(
            buffer_size=buffer_size,
            state_shape=(self.state_dim,),
            action_shape=(self.action_dim,),
            mix=1
        )

        # RGCL scan step with PPO updates
        def scan_step(carry, t):
            state, theta, P, key, policy_params, value_params, buffer = carry

            # Unflatten theta → params
            p_list = []
            idx = 0
            for p in flat_params:
                size = p.size
                p_list.append(theta[idx:idx+size].reshape(p.shape))
                idx += size
            local_params = jax.tree_util.tree_unflatten(treedef, p_list)

            # Update cost function training state with current params
            state_train_local = state_train.replace(params=local_params)

            # Sample action from PPO policy (using current policy params)
            key, subkey = jax.random.split(key)
            mu, log_std = self.policy_net.apply(
                {'params': policy_params},
                state[None, :]
            )
            noise = jax.random.normal(subkey, mu.shape)
            u = mu + jnp.exp(log_std) * noise
            action = jnp.tanh(u).flatten()
            logp = self.calculate_log_pi(log_std, noise, jnp.tanh(u))

            # Environment transition
            next_state = self._dynamics(state, action)
            next_state = jnp.atleast_1d(next_state).flatten()

            # Compute environment reward
            env_reward = self.reward_fn(state, action, next_state)

            # Compute learned cost (use as negative reward for PPO)
            learned_cost = state_train_local.apply_fn(
                {'params': local_params},
                state.reshape(1, -1)
            ).flatten()[0]
            ppo_reward = -learned_cost  # PPO maximizes reward = minimizes cost

            # Add transition to buffer
            done = jnp.array([0.0])
            buffer = buffer.append(
                state, action, jnp.array([ppo_reward]),
                done, logp.flatten(), next_state
            )

            # PPO Update: Every ppo_update_freq steps, if buffer has enough samples
            def do_ppo_update(policy_params, value_params, buffer):
                """Perform PPO update using buffer samples"""
                # Get samples from buffer (last buffer_size samples)
                # Use direct indexing instead of buffer.get() to avoid assertion issues in JAX tracing
                n_samples = jnp.minimum(buffer.n, buffer.buffer_size)
                end_idx = buffer.p
                start_idx = (end_idx - n_samples) % buffer.total_size

                # Handle wraparound
                indices = jnp.arange(start_idx, start_idx + n_samples) % buffer.total_size
                buffer_states = buffer.states[indices]
                buffer_actions = buffer.actions[indices]
                buffer_rewards = buffer.rewards[indices]
                buffer_dones = buffer.dones[indices]
                buffer_logps = buffer.log_pis[indices]
                buffer_next_states = buffer.next_states[indices]

                # Compute values
                values = self.value_fn.apply_fn({'params': value_params}, buffer_states).flatten()
                next_values = self.value_fn.apply_fn({'params': value_params}, buffer_next_states).flatten()

                # Compute GAE advantages
                deltas = buffer_rewards.flatten() + gamma * (1.0 - buffer_dones.flatten()) * next_values - values
                advantages = jnp.zeros_like(deltas)
                gae = 0.0
                for i in range(len(deltas) - 1, -1, -1):
                    gae = deltas[i] + gamma * lam * (1.0 - buffer_dones.flatten()[i]) * gae
                    advantages = advantages.at[i].set(gae)

                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Actor loss
                def actor_loss_fn(params):
                    new_logp = self.evaluate_log_pi(buffer_states, buffer_actions, params)
                    ratio = jnp.exp(new_logp - buffer_logps.flatten())
                    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                    loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
                    return loss

                # Critic loss
                def critic_loss_fn(params):
                    v = self.value_fn.apply_fn({'params': params}, buffer_states).flatten()
                    return jnp.mean((returns - v) ** 2)

                # Update policy
                policy_grads = jax.grad(actor_loss_fn)(policy_params)
                policy_params_new = jax.tree_util.tree_map(
                    lambda p, g: p - 3e-4 * g, policy_params, policy_grads
                )

                # Update value function
                value_grads = jax.grad(critic_loss_fn)(value_params)
                value_params_new = jax.tree_util.tree_map(
                    lambda p, g: p - 3e-4 * g, value_params, value_grads
                )

                return policy_params_new, value_params_new

            # Conditionally update PPO
            should_update = (t % ppo_update_freq == 0) & (t > 0) & (buffer.n >= buffer_size)
            policy_params_new, value_params_new = jax.lax.cond(
                should_update,
                do_ppo_update,
                lambda p, v, b: (p, v),  # No update
                policy_params, value_params, buffer
            )

            # Compute gradients for RGCL update
            g_s = get_gradients(state_train_local, local_params, next_state, args.N_steps)
            g_d = get_gradients(state_train_local, local_params, expert_states[t], args.N_steps)

            # Compute Hessians for RGCL update
            H_s = get_hessian(state_train_local, local_params, next_state, args.N_steps)
            H_d = get_hessian(state_train_local, local_params, expert_states[t], args.N_steps)

            # Kalman-style parameter update
            P_new = jnp.linalg.inv(jnp.linalg.inv(P + Q) + (H_d - H_s))
            theta_new = theta - P_new @ (g_d - g_s)
            theta_new = theta_new.astype(jnp.float32)

            traj_prob = logp.flatten()[0]

            return (next_state, theta_new, P_new, key, policy_params_new, value_params_new, buffer), \
                   (state, traj_prob, action, env_reward, P_new)

        # Execute RGCL scan over all timesteps
        init_carry = (init_state, theta0, P0, key,
                     self.policy_model.params, self.value_fn.params, experience_buffer)

        (final_state, final_theta, final_P, _, final_policy_params, final_value_params, _), \
        (states, traj_probs, actions, rewards, P_theta) = lax.scan(
            scan_step,
            init_carry,
            jnp.arange(args.N_steps)
        )

        # Update policy model with final params
        self.policy_model = self.policy_model.replace(params=final_policy_params)
        self.value_fn = self.value_fn.replace(params=final_value_params)

        # Unflatten final theta back into params
        idx = 0
        new_param_list = []
        for p in flat_params:
            size = p.size
            new_param_list.append(final_theta[idx:idx+size].reshape(p.shape))
            idx += size
        new_params = jax.tree_util.tree_unflatten(treedef, new_param_list)
        params = new_params

        # Compute total reward
        total_reward = jnp.sum(rewards)

        # Convert to lists for compatibility
        states, traj_probs, actions, total_reward = (
            states.tolist(),
            traj_probs.tolist(),
            actions.tolist(),
            total_reward.tolist()
        )

        return states, traj_probs, actions, total_reward, P_theta, params

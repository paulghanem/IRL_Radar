"""
Optimized MPPI class with JIT compilation and performance improvements.

Key optimizations:
1. JIT-compiled forward_pure method
2. Static argument annotations for frame_skip, gym_env
3. Removed unnecessary .tolist() conversions
4. Reduced CPU-GPU transfers
5. Closure-based access to JIT-compiled functions to avoid nested JIT issues
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

from src.control.dynamics import kinematics_mujoco


def create_optimized_generate_session(mppi_instance):
    """
    Creates a JIT-compiled session generator for the MPPI instance.

    This wraps lax.scan with all static arguments pre-bound.
    Captures cost_func and reward_fn as closures to avoid passing them through lax.scan.
    """

    # Capture JIT-compiled functions as closure variables
    cost_func = mppi_instance._cost_func
    reward_fn = mppi_instance.reward_fn

    # Capture all MPPI parameters as closure variables
    u_min = mppi_instance._u_min
    u_max = mppi_instance._u_max
    mjx_model = mppi_instance.mjx_model
    mjx_data = mppi_instance.mjx_data
    num_samples = mppi_instance._num_samples
    horizon = mppi_instance._horizon
    dim_state = mppi_instance._dim_state
    dim_control = mppi_instance._dim_control
    exploration = mppi_instance._exploration
    lambda_ = mppi_instance._lambda

    # Extract diagonal elements from covariance for noise generation
    # mppi_instance._covariance is shape (horizon, dim_control, dim_control)
    # We need the standard deviations for each action dimension
    # Since covariance is diagonal, extract diagonal elements: shape (horizon, dim_control)
    covariance_diag = jnp.array([jnp.diag(mppi_instance._covariance[h]) for h in range(horizon)])
    std_devs = jnp.sqrt(covariance_diag)  # Shape: (horizon, dim_control)
    sample_shape = (num_samples, horizon, dim_control)

    # Define forward_pure_optimized INSIDE to access cost_func via closure
    # Do NOT add @jax.jit here - it will be traced as part of outer JIT compilation
    def forward_pure_optimized(
        state,
        key,
        prev_action_seq,
        state_train,
        gym_env,
        frame_skip,
        gail=False
    ):
        """
        Pure MPPI forward step with JIT compilation.

        Accesses cost_func through closure to avoid passing JIT-compiled functions as parameters.

        Args:
            state: jnp.ndarray shape (state_dim,)
            key: PRNGKey
            prev_action_seq: jnp.ndarray shape (horizon, action_dim)
            state_train: training state for cost function
            gym_env: string, environment name (static)
            frame_skip: int, frame skip (static)
            gail: bool, whether to use GAIL (static)

        Returns:
            optimal_action_seq: (horizon, action_dim)
            new_key: PRNGKey
            new_prev_action_seq: (horizon, action_dim)
        """

        mean_action_seq = prev_action_seq

        # Random sampling with reparametrization trick
        # Generate Gaussian noise for each action dimension
        key, subkey = jax.random.split(key)
        action_noises = jax.random.normal(subkey, shape=sample_shape) * std_devs

        # Noise injection with exploration
        threshold = int(num_samples * (1.0 - exploration))
        inherited_samples = mean_action_seq + action_noises[:threshold]
        perturbed_action_seqs = jnp.concatenate(
            [inherited_samples, action_noises[threshold:]], axis=0
        )

        # Clamp actions
        perturbed_action_seqs = jnp.clip(
            perturbed_action_seqs, u_min, u_max
        )

        # Rollout samples in parallel
        st = state
        if gym_env in ["Ant"]:
            st = jnp.concatenate((jnp.reshape(mjx_data.qpos[0:2], (2,)), st))
        elif gym_env in ["Walker2d-v4", "Walker2d", "Hopper-v4", "Hopper", "HalfCheetah-v4"]:
            # For these envs, observation excludes x-position, so prepend it
            st = jnp.concatenate((jnp.reshape(mjx_data.qpos[0:1], (1,)), st))

        initial_state = jnp.tile(st, (num_samples, 1))

        state_seq_batch = jax.vmap(
            kinematics_mujoco, in_axes=(None, None, 0, 0, None, None)
        )(mjx_model, mjx_data, initial_state, perturbed_action_seqs, gym_env, frame_skip)

        initial_state_reshaped = initial_state.reshape((initial_state.shape[0], 1, initial_state.shape[1]))
        state_seq_batch = jnp.concatenate((initial_state_reshaped, state_seq_batch), axis=1)

        if gym_env in ["Ant"]:
            state_seq_batch = state_seq_batch[:, :, 2:]

        # Compute sample costs - cost_func accessed via closure!
        costs = jax.vmap(cost_func, in_axes=(1, None))(state_seq_batch[:, :-1, :], state_train)
        costs = costs[:, :, 0]
        costs = costs.T

        terminal_costs = cost_func(
            state_seq_batch[:, -1, :], state_train
        ).ravel()

        total_costs = jnp.sum(costs, axis=1) + terminal_costs

        if gail:
            D = jnp.exp(-total_costs) / (jnp.exp(-total_costs) + 1.0)
            total_costs = -jnp.log(D)

        # Weights and optimal control
        weights = jax.nn.softmax(-total_costs / lambda_, axis=0)
        optimal_action_seq = jnp.sum(
            weights.reshape(num_samples, 1, 1) * perturbed_action_seqs,
            axis=0,
        )

        new_prev_action_seq = optimal_action_seq

        return optimal_action_seq, key, new_prev_action_seq

    # Define rollout_step inside this function to access forward_pure_optimized and reward_fn via closure
    def rollout_step_optimized(
        carry,
        t,
        gym_env,
        frame_skip,
        dt,
        gail
    ):
        """
        Rollout step for lax.scan - traced as part of outer JIT compilation.

        Accesses cost_func and reward_fn through closure to avoid passing
        JIT-compiled functions as parameters (which JAX cannot handle in lax.scan).
        """
        state, key, prev_action_seq, state_train = carry

        # Forward pass - accesses cost_func via closure
        action_seq, key, prev_action_seq = forward_pure_optimized(
            state=state,
            key=key,
            prev_action_seq=prev_action_seq,
            state_train=state_train,
            gym_env=gym_env,
            frame_skip=frame_skip,
            gail=gail
        )

        # Dynamics update
        next_state = kinematics_mujoco(
            mjx_model, mjx_data, state.flatten(),
            action_seq[0, :].reshape((1, -1)), gym_env, frame_skip=frame_skip
        ).flatten()

        action = action_seq[0, :]

        # Compute reward - reward_fn accessed via closure
        r = reward_fn(gym_env, state, action, next_state, mjx_data, dt, frame_skip)

        new_carry = (next_state, key, prev_action_seq, state_train)
        outputs = (state, action, r)

        return new_carry, outputs

    @partial(
        jax.jit,
        static_argnames=('N_steps', 'gym_env', 'frame_skip', 'gail')
    )
    def generate_session_optimized(
        init_state,
        key,
        prev_action_seq,
        state_train,
        N_steps,
        gym_env,
        frame_skip,
        dt,
        gail=False
    ):
        """
        Fully JIT-compiled session generation using lax.scan.

        Returns:
            states: (N_steps, state_dim)
            actions: (N_steps, action_dim)
            total_reward: scalar
            final_prev_action_seq: (horizon, action_dim)
        """

        (final_state, final_key, final_prev_action_seq, _), (states, actions, rewards) = jax.lax.scan(
            lambda carry, t: rollout_step_optimized(
                carry, t,
                gym_env, frame_skip, dt, gail
            ),
            (init_state, key, prev_action_seq, state_train),
            jnp.arange(N_steps)
        )

        total_reward = jnp.sum(rewards)

        return states, actions, total_reward, final_prev_action_seq

    return generate_session_optimized

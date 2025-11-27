#!/usr/bin/env python3
"""
Test Simplified Walker2d (Pure JAX) vs MJX
Tests both GCL and RGCL LAX methods
Parameters: horizon=50, num_traj=500, N_steps=1000, iterations=1000
Reports: rewards and execution time
"""
import os
import sys
import argparse

# Get platform from command line or default to cpu
parser = argparse.ArgumentParser()
parser.add_argument('--platform', type=str, default='cpu', choices=['cpu', 'cuda'])
platform_args, remaining_args = parser.parse_known_args()

# Set JAX platform BEFORE any JAX imports
os.environ['JAX_PLATFORMS'] = platform_args.platform
# If using CPU, hide CUDA devices to prevent JAX from trying to initialize them
if platform_args.platform == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import time
import jax
import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--method', type=str, default='gcl', choices=['gcl', 'rgcl'])
    parser.add_argument('--gym_env', type=str, default='SimpleWalker2d')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--num_traj', type=int, default=500)
    parser.add_argument('--N_steps', type=int, default=1000)
    parser.add_argument('--N_steps_expert', type=int, default=1000)
    parser.add_argument('--rirl_iterations', type=int, default=1000)
    parser.add_argument('--reward_fn_updates', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_', type=float, default=0.01)
    parser.add_argument('--Q', type=float, default=1e-5)
    parser.add_argument('--P', type=float, default=1e-2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--s_dim', type=int, default=18)
    parser.add_argument('--a_dim', type=int, default=6)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--frame_skip', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--gail', action='store_true', default=False)
    parser.add_argument('--rgcl', action='store_true', default=False)
    parser.add_argument('--no-save_images', action='store_true', default=True)
    parser.add_argument('--diagonal', action='store_true', default=False)

    args = parser.parse_args()

    # Set rgcl flag based on method
    if args.method == 'rgcl':
        args.rgcl = True

    print("=" * 80)
    print(f"SIMPLIFIED WALKER2D TEST - {args.method.upper()}")
    print("=" * 80)
    print(f"Method: {args.method.upper()}")
    print(f"Horizon: {args.horizon}, Trajectories: {args.num_traj}")
    print(f"N_steps: {args.N_steps}, Iterations: {args.rirl_iterations}")
    print(f"Q: {args.Q}, P: {args.P}")
    print("=" * 80)

    # Import after setting JAX platform
    from src.control.mppi_class import MPPI
    from src.control.simple_walker2d import (
        simple_walker2d_step,
        simple_walker2d_reward,
        simple_walker2d_reset,
        get_simple_walker2d_params
    )

    # Get environment parameters
    env_params = get_simple_walker2d_params()
    s_dim = env_params['state_dim']
    a_dim = env_params['action_dim']

    # Create MPPI controller with simple Walker2d dynamics
    u_min = env_params['action_min']
    u_max = env_params['action_max']
    sigmas = jnp.array([args.sigma] * a_dim)

    # Cost function for MPPI - direct computation based on Walker2d physics
    def cost_func(state, state_train):
        """
        Compute costs directly from state to guide MPPI trajectory optimization.
        This provides clear gradients for forward movement.
        """
        # Extract state components (SimpleWalker2d has 18-dim state)
        x_pos = state[:, 0]  # Forward position
        z_height = state[:, 1]  # Height
        body_angle = state[:, 2]  # Body angle
        x_vel = state[:, 8] if state.shape[1] > 8 else 0.0  # Forward velocity (if available)

        # Cost = negative reward for MPPI minimization
        # Reward components from simple_walker2d_reward:
        # 1. Forward velocity (primary objective)
        forward_cost = -x_pos  # Encourage higher x position

        # 2. Survival penalty (stay upright)
        fall_penalty = jnp.where(
            (jnp.abs(body_angle) > 1.0) | (z_height < 0.8) | (z_height > 2.0),
            100.0,  # Large penalty for falling
            0.0
        )

        # 3. Balance cost (small penalty for tilting)
        balance_cost = 0.5 * jnp.square(body_angle)

        # Total cost
        costs = forward_cost + fall_penalty + balance_cost

        return costs.reshape(-1, 1)

    # Create dummy state_train for initialization
    class DummyStateTrain:
        def __init__(self):
            self.params = None

    state_train = DummyStateTrain()

    # Create minimal env wrapper for reset functionality
    class SimpleWalker2dEnv:
        def reset(self, seed=None):
            return simple_walker2d_reset(), {}

    simple_env = SimpleWalker2dEnv()

    # Custom dynamics wrapper for simple Walker2d
    def simple_dynamics(state, action):
        """Wrapper for MPPI compatibility"""
        return simple_walker2d_step(state, action)

    mppi = MPPI(
        state_train=state_train,
        horizon=args.horizon,
        num_samples=args.num_traj,
        dim_state=s_dim,
        dim_control=a_dim,
        dynamics=simple_dynamics,
        cost_func=cost_func,
        u_min=u_min,
        u_max=u_max,
        sigmas=sigmas,
        lambda_=args.lambda_,
        exploration=0.0,
        seed=args.seed,
        env=simple_env,
        mjx_model=None,  # No MJX for simple Walker2d
        gym_env='SimpleWalker2d',
        env_brax=None,
        use_mujoco=False  # Pure JAX dynamics
    )

    # Override reward function to use simple Walker2d reward
    original_reward_fn = mppi.reward_fn
    def simple_reward_fn(gym_env, state, action, next_state, mjx_data, dt, frame_skip):
        return simple_walker2d_reward(state, action, next_state)
    mppi.reward_fn = simple_reward_fn

    # Generate expert demonstration data using forward walking policy
    print("Generating expert demonstrations...")
    import numpy as np
    D_demo = np.zeros((args.N_steps, s_dim + a_dim))
    state = simple_walker2d_reset()

    # Generate expert trajectory with forward bias
    for i in range(args.N_steps):
        # Forward biased action to encourage walking
        action = jnp.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
        D_demo[i, :s_dim] = state
        D_demo[i, s_dim:] = action
        state = simple_walker2d_step(state, action)

    D_demo = jnp.array(D_demo)

    # Calculate expert reward
    expert_reward = 0.0
    for i in range(args.N_steps-1):
        r = simple_walker2d_reward(D_demo[i,:s_dim], D_demo[i,s_dim:], D_demo[i+1,:s_dim])
        expert_reward += float(r)
    print(f"Expert trajectory generated. Total expert reward: {expert_reward:.2f}")

    # Initialize state_train with proper parameters
    print("Initializing reward model...")
    from cost_jax import CostNN, apply_model, update_model
    from flax.training import train_state
    import optax

    key = jax.random.PRNGKey(args.seed)
    cost_f = CostNN(state_dims=s_dim, hidden_dim=args.hidden_dim)
    init_rng = jax.random.PRNGKey(args.seed)
    variables = cost_f.init(init_rng, jnp.ones((1, s_dim)))
    params = variables['params']
    tx = optax.adam(learning_rate=args.lr)
    state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)
    mppi.state_train = state_train

    # Helper function to preprocess trajectory
    def preprocess_traj(trajs):
        states, probs, actions = trajs[0][0], trajs[0][1], trajs[0][2]
        D_samp = np.column_stack([states, probs, actions])
        return jnp.array(D_samp)

    print("\n" + "=" * 80)
    print(f"RUNNING {args.method.upper()} WITH SIMPLIFIED WALKER2D")
    print("=" * 80)

    if args.method == 'gcl':
        # Test GCL method
        print("Testing GCL with generate_session_lax...")

        # Warmup run (JIT compilation)
        print("Warmup run (includes JIT compilation)...")
        start_warmup = time.time()
        states_warm, probs_warm, actions_warm, rewards_warm = mppi.generate_session_lax(
            args, state_train, D_demo, mpc_method=None, thetas=None
        )
        jax.block_until_ready(rewards_warm)
        warmup_time = time.time() - start_warmup
        print(f"Warmup completed: {warmup_time:.2f} seconds")

        # Main experiment with GCL learning
        print(f"\nRunning {args.rirl_iterations} iterations with reward function updates...")
        iteration_times = []
        iteration_rewards = []
        reward_losses = []

        start_total = time.time()
        for iteration in range(args.rirl_iterations):
            iter_start = time.time()

            # Generate policy rollout
            states, probs, actions, rewards = mppi.generate_session_lax(
                args, state_train, D_demo, mpc_method=None, thetas=None
            )
            jax.block_until_ready(rewards)

            # Compute TRUE rewards for the rollout
            # Convert to JAX arrays if needed
            states_arr = jnp.array(states) if not isinstance(states, jnp.ndarray) else states
            actions_arr = jnp.array(actions) if not isinstance(actions, jnp.ndarray) else actions

            true_rewards = []
            for i in range(len(states_arr)-1):
                r = simple_walker2d_reward(states_arr[i], actions_arr[i], states_arr[i+1])
                true_rewards.append(float(r))
            true_rewards = jnp.array(true_rewards)

            # Update reward function to predict TRUE rewards (supervised learning)
            loss_rew = []
            for _ in range(args.reward_fn_updates):
                # Train on current rollout states to predict true rewards
                def loss_fn(params):
                    predicted = state_train.apply_fn({'params': params}, states_arr[:-1])
                    mse_loss = jnp.mean(jnp.square(predicted.squeeze() - true_rewards))
                    return mse_loss

                loss_val, grads = jax.value_and_grad(loss_fn)(state_train.params)
                state_train = state_train.apply_gradients(grads=grads)
                loss_rew.append(float(loss_val))

            # Update MPPI with new reward function
            mppi.state_train = state_train

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            iteration_rewards.append(float(rewards))
            reward_losses.append(np.mean(loss_rew))

            if (iteration + 1) % 100 == 0:
                avg_time = sum(iteration_times[-100:]) / 100
                avg_reward = sum(iteration_rewards[-100:]) / 100
                avg_loss = np.mean(reward_losses[-100:])
                print(f"Iteration {iteration + 1}/{args.rirl_iterations}: "
                      f"Avg Time={avg_time:.4f}s, Avg Reward={avg_reward:.2f}, Avg Loss={avg_loss:.4f}")

        total_time = time.time() - start_total

    else:  # rgcl
        # Test RGCL LAX method
        print("Testing RGCL with RGCL_lax...")

        # Initialize RGCL parameters
        if args.diagonal:
            params = jnp.ones(s_dim + a_dim)
        else:
            params = jnp.eye(s_dim + a_dim).flatten()

        P_theta = jnp.eye(len(params)) * args.P
        initial_state = simple_walker2d_reset()

        # Warmup run
        print("Warmup run (includes JIT compilation)...")
        start_warmup = time.time()
        _, _, _, _, rewards_warm = mppi.RGCL_lax(
            args, params, state_train, initial_state, D_demo, P_theta, thetas=None
        )
        jax.block_until_ready(rewards_warm)
        warmup_time = time.time() - start_warmup
        print(f"Warmup completed: {warmup_time:.2f} seconds")

        # Main experiment
        print(f"\nRunning {args.rirl_iterations} iterations...")
        iteration_times = []
        iteration_rewards = []

        start_total = time.time()
        for iteration in range(args.rirl_iterations):
            iter_start = time.time()

            params, P_theta, states, actions, rewards = mppi.RGCL_lax(
                args, params, state_train, initial_state, D_demo, P_theta, thetas=None
            )
            jax.block_until_ready(rewards)

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            iteration_rewards.append(float(rewards))

            if (iteration + 1) % 100 == 0:
                avg_time = sum(iteration_times[-100:]) / 100
                avg_reward = sum(iteration_rewards[-100:]) / 100
                print(f"Iteration {iteration + 1}/{args.rirl_iterations}: "
                      f"Avg Time={avg_time:.4f}s, Avg Reward={avg_reward:.2f}")

        total_time = time.time() - start_total

    # Calculate statistics
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    avg_reward = sum(iteration_rewards) / len(iteration_rewards)
    final_reward = sum(iteration_rewards[-100:]) / 100  # Last 100 iterations

    print("\n" + "=" * 80)
    print(f"RESULTS - {args.method.upper()} WITH SIMPLIFIED WALKER2D")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Average time per iteration: {avg_iter_time:.4f} seconds")
    print(f"Average reward (all iterations): {avg_reward:.2f}")
    print(f"Final reward (last 100 iterations): {final_reward:.2f}")
    print(f"Time per step (avg): {avg_iter_time / args.N_steps:.4f} seconds")
    print("=" * 80)

    # Save results to file
    results_file = f"simple_walker2d_{args.method}_seed{args.seed}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Method: {args.method.upper()}\n")
        f.write(f"Parameters: horizon={args.horizon}, num_traj={args.num_traj}\n")
        f.write(f"N_steps={args.N_steps}, iterations={args.rirl_iterations}\n")
        f.write(f"Q={args.Q}, P={args.P}\n\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n")
        f.write(f"Average time per iteration: {avg_iter_time:.4f} seconds\n")
        f.write(f"Average reward (all iterations): {avg_reward:.2f}\n")
        f.write(f"Final reward (last 100 iterations): {final_reward:.2f}\n")
        f.write(f"Time per step (avg): {avg_iter_time / args.N_steps:.4f} seconds\n")
        f.write("\nIteration-by-iteration data:\n")
        for i, (t, r) in enumerate(zip(iteration_times, iteration_rewards)):
            f.write(f"{i+1},{t:.4f},{r:.2f}\n")

    print(f"\nResults saved to {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

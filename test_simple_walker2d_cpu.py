#!/usr/bin/env python3
"""
Quick CPU test for Simplified Walker2d
Small parameters to verify code works before GPU run
"""
import time
import jax
import jax.numpy as jnp
import argparse
import sys
import os

# Force JAX to use CPU for testing
os.environ['JAX_PLATFORMS'] = 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gcl', choices=['gcl', 'rgcl'])
    parser.add_argument('--gym_env', type=str, default='SimpleWalker2d')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--horizon', type=int, default=5)  # Small for CPU test
    parser.add_argument('--num_traj', type=int, default=50)  # Small for CPU test
    parser.add_argument('--N_steps', type=int, default=10)  # Small for CPU test
    parser.add_argument('--N_steps_expert', type=int, default=10)
    parser.add_argument('--rirl_iterations', type=int, default=3)  # Just 3 iterations to test
    parser.add_argument('--reward_fn_updates', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_', type=float, default=0.01)
    parser.add_argument('--Q', type=float, default=1e-5)
    parser.add_argument('--P', type=float, default=1e-2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--s_dim', type=int, default=17)
    parser.add_argument('--a_dim', type=int, default=6)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--frame_skip', type=int, default=5)
    parser.add_argument('--gail', action='store_true', default=False)
    parser.add_argument('--rgcl', action='store_true', default=False)
    parser.add_argument('--no-save_images', action='store_true', default=True)
    parser.add_argument('--diagonal', action='store_true', default=False)

    args = parser.parse_args()

    # Set rgcl flag based on method
    if args.method == 'rgcl':
        args.rgcl = True

    print("=" * 80)
    print(f"CPU TEST - SIMPLIFIED WALKER2D - {args.method.upper()}")
    print("=" * 80)
    print(f"Method: {args.method.upper()}")
    print(f"Horizon: {args.horizon}, Trajectories: {args.num_traj}")
    print(f"N_steps: {args.N_steps}, Iterations: {args.rirl_iterations}")
    print(f"Platform: CPU (testing)")
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

    print(f"\nEnvironment: state_dim={s_dim}, action_dim={a_dim}")

    # Create MPPI controller with simple Walker2d dynamics
    u_min = env_params['action_min']
    u_max = env_params['action_max']
    sigmas = jnp.array([1.0] * a_dim)

    # Dummy cost function (not used in IRL)
    def cost_func(state, state_train):
        return jnp.zeros((state.shape[0], 1))

    # Create dummy state_train for initialization
    class DummyStateTrain:
        def __init__(self):
            self.params = None

    state_train = DummyStateTrain()

    # Custom dynamics wrapper for simple Walker2d
    def simple_dynamics(state, action):
        """Wrapper for MPPI compatibility"""
        return simple_walker2d_step(state, action)

    print("\nCreating MPPI controller...")
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
        env=None,
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

    print("Creating demo data...")
    # Create dummy demo data
    D_demo = jnp.zeros((args.N_steps, s_dim + a_dim))
    # Initialize with reasonable values
    for i in range(args.N_steps):
        D_demo = D_demo.at[i, :s_dim].set(simple_walker2d_reset())

    # Initialize state_train with proper parameters
    print("Initializing reward model...")
    from utils.helpers import StateModel_flax
    key = jax.random.PRNGKey(args.seed)
    state_train = StateModel_flax(
        key=key,
        n_actions=a_dim,
        n_states=s_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1
    )
    mppi.state_train = state_train

    print("\n" + "=" * 80)
    print(f"RUNNING {args.method.upper()} TEST")
    print("=" * 80)

    if args.method == 'gcl':
        # Test GCL method
        print("Testing GCL with generate_session_lax...")

        # First run (JIT compilation)
        print("\nRun 1 (includes JIT compilation)...")
        start1 = time.time()
        states1, probs1, actions1, rewards1 = mppi.generate_session_lax(
            args, state_train, D_demo, mpc_method=None, thetas=None
        )
        jax.block_until_ready(rewards1)
        time1 = time.time() - start1
        print(f"Run 1: {time1:.2f} seconds, Reward: {rewards1:.2f}")

        # Second run (cached)
        print("\nRun 2 (JIT cached)...")
        start2 = time.time()
        states2, probs2, actions2, rewards2 = mppi.generate_session_lax(
            args, state_train, D_demo, mpc_method=None, thetas=None
        )
        jax.block_until_ready(rewards2)
        time2 = time.time() - start2
        print(f"Run 2: {time2:.2f} seconds, Reward: {rewards2:.2f}")

        # Third run
        print("\nRun 3 (JIT cached)...")
        start3 = time.time()
        states3, probs3, actions3, rewards3 = mppi.generate_session_lax(
            args, state_train, D_demo, mpc_method=None, thetas=None
        )
        jax.block_until_ready(rewards3)
        time3 = time.time() - start3
        print(f"Run 3: {time3:.2f} seconds, Reward: {rewards3:.2f}")

        avg_time = (time2 + time3) / 2
        print(f"\nAverage time (cached): {avg_time:.2f} seconds")
        print(f"Time per step: {avg_time / args.N_steps:.4f} seconds")

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

        # First run
        print("\nRun 1 (includes JIT compilation)...")
        start1 = time.time()
        params1, P_theta1, states1, actions1, rewards1 = mppi.RGCL_lax(
            args, params, state_train, initial_state, D_demo, P_theta, thetas=None
        )
        jax.block_until_ready(rewards1)
        time1 = time.time() - start1
        print(f"Run 1: {time1:.2f} seconds, Reward: {rewards1:.2f}")

        # Second run
        print("\nRun 2 (JIT cached)...")
        start2 = time.time()
        params2, P_theta2, states2, actions2, rewards2 = mppi.RGCL_lax(
            args, params1, state_train, initial_state, D_demo, P_theta1, thetas=None
        )
        jax.block_until_ready(rewards2)
        time2 = time.time() - start2
        print(f"Run 2: {time2:.2f} seconds, Reward: {rewards2:.2f}")

        # Third run
        print("\nRun 3 (JIT cached)...")
        start3 = time.time()
        params3, P_theta3, states3, actions3, rewards3 = mppi.RGCL_lax(
            args, params2, state_train, initial_state, D_demo, P_theta2, thetas=None
        )
        jax.block_until_ready(rewards3)
        time3 = time.time() - start3
        print(f"Run 3: {time3:.2f} seconds, Reward: {rewards3:.2f}")

        avg_time = (time2 + time3) / 2
        print(f"\nAverage time (cached): {avg_time:.2f} seconds")
        print(f"Time per step: {avg_time / args.N_steps:.4f} seconds")

    print("\n" + "=" * 80)
    print("CPU TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nCode is working! Ready for GPU run with full parameters.")
    print(f"Estimated GPU time for full run (horizon=50, num_traj=500, 1000 steps, 1000 iter):")
    time_per_step_gpu_estimate = avg_time / args.N_steps * (50/args.horizon) * (500/args.num_traj) / 10  # GPU ~10x faster
    print(f"  Per step: ~{time_per_step_gpu_estimate:.4f} seconds")
    print(f"  1000 iterations: ~{time_per_step_gpu_estimate * 1000 * 1000 / 3600:.2f} hours")
    print("=" * 80)

if __name__ == "__main__":
    main()

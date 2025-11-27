#!/usr/bin/env python3
"""
Test script to verify 10x speedup from JIT optimizations
"""
import time
import jax
import jax.numpy as jnp
import argparse
import sys

# Force JAX to use GPU
import os
os.environ['JAX_PLATFORMS'] = 'cuda'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_env', type=str, default='Walker2d')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--num_traj', type=int, default=500)
    parser.add_argument('--N_steps', type=int, default=10)  # Short test
    parser.add_argument('--N_steps_expert', type=int, default=1000)
    parser.add_argument('--rirl_iterations', type=int, default=1)
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

    print("=" * 80)
    print("SPEED OPTIMIZATION TEST")
    print("=" * 80)
    print(f"Environment: {args.gym_env}")
    print(f"Horizon: {args.horizon}, Trajectories: {args.num_traj}")
    print(f"N_steps (test): {args.N_steps}")
    print("=" * 80)

    # Import after setting JAX platform
    from src.control.mppi_class import MPPI
    from mujoco import mjx
    import mujoco as mj

    # Load MuJoCo model
    if args.gym_env == "Walker2d":
        xml_path = "/ocean/projects/cis250114p/pghanem/IRL_Radar_big/mujoco_menagerie/walker2d/walker2d.xml"
    else:
        print(f"Error: Environment {args.gym_env} not supported in this test")
        return

    mj_model = mj.MjModel.from_xml_path(xml_path)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # Get dimensions
    s_dim = args.s_dim
    a_dim = args.a_dim

    # Create MPPI controller
    u_min = jnp.array([-1.0] * a_dim)
    u_max = jnp.array([1.0] * a_dim)
    sigmas = jnp.array([1.0] * a_dim)

    # Dummy cost function
    def cost_func(state, state_train):
        return jnp.zeros((state.shape[0], 1))

    # Create dummy state_train
    class DummyStateTrain:
        def __init__(self):
            self.params = None

    state_train = DummyStateTrain()

    mppi = MPPI(
        state_train=state_train,
        horizon=args.horizon,
        num_samples=args.num_traj,
        dim_state=s_dim,
        dim_control=a_dim,
        dynamics=None,
        cost_func=cost_func,
        u_min=u_min,
        u_max=u_max,
        sigmas=sigmas,
        lambda_=args.lambda_,
        exploration=0.0,
        seed=args.seed,
        env=None,
        mjx_model=mjx_model,
        gym_env=args.gym_env,
        env_brax=None,
        use_mujoco=True
    )

    # Create dummy demo data
    D_demo = jnp.zeros((args.N_steps, s_dim + a_dim))
    D_demo = D_demo.at[:, :s_dim].set(jnp.ones((args.N_steps, s_dim)) * 0.1)

    print("\n" + "=" * 80)
    print("RUNNING SPEED TEST")
    print("=" * 80)
    print("First call will include JIT compilation time...")

    # First call (includes compilation)
    start_compile = time.time()
    states1, probs1, actions1, rewards1 = mppi.generate_session_lax_jit(
        args, state_train, D_demo, mpc_method=None, thetas=None
    )
    jax.block_until_ready(rewards1)
    compile_time = time.time() - start_compile

    print(f"First call (with compilation): {compile_time:.4f} seconds")
    print(f"Total reward: {rewards1:.2f}")

    # Second call (pure execution)
    print("\nSecond call (pure execution, JIT cached)...")
    start_exec = time.time()
    states2, probs2, actions2, rewards2 = mppi.generate_session_lax_jit(
        args, state_train, D_demo, mpc_method=None, thetas=None
    )
    jax.block_until_ready(rewards2)
    exec_time = time.time() - start_exec

    print(f"Second call (cached): {exec_time:.4f} seconds")
    print(f"Total reward: {rewards2:.2f}")

    # Third call for consistency
    start_exec2 = time.time()
    states3, probs3, actions3, rewards3 = mppi.generate_session_lax_jit(
        args, state_train, D_demo, mpc_method=None, thetas=None
    )
    jax.block_until_ready(rewards3)
    exec_time2 = time.time() - start_exec2

    print(f"Third call (cached): {exec_time2:.4f} seconds")
    print(f"Total reward: {rewards3:.2f}")

    avg_exec = (exec_time + exec_time2) / 2

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average execution time (cached): {avg_exec:.4f} seconds")
    print(f"Time per step: {avg_exec / args.N_steps:.4f} seconds")
    print(f"Estimated time for 100 steps: {(avg_exec / args.N_steps) * 100:.2f} seconds")
    print(f"Estimated time for 1000 steps: {(avg_exec / args.N_steps) * 1000:.2f} seconds")
    print(f"Estimated time for 1000 iterations (1M steps): {(avg_exec / args.N_steps) * 1000000 / 3600:.2f} hours")
    print("=" * 80)

    # Calculate expected speedup
    baseline_per_step = 5.0  # seconds (from previous measurements)
    optimized_per_step = avg_exec / args.N_steps
    speedup = baseline_per_step / optimized_per_step

    print(f"\nESTIMATED SPEEDUP:")
    print(f"Baseline: {baseline_per_step:.2f} sec/step")
    print(f"Optimized: {optimized_per_step:.4f} sec/step")
    print(f"Speedup: {speedup:.1f}x")
    print("=" * 80)

if __name__ == "__main__":
    main()

# %%
# Configure CUDA library paths for JAX - must be done BEFORE importing JAX
import os
import sys
import glob

# Set JAX environment variables
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Find and configure CUDA library paths from site-packages
site_packages = None
for path in sys.path:
    if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
        site_packages = path
        break

if site_packages:
    nvidia_path = os.path.join(site_packages, 'nvidia')
    subdirs = [d for d in os.listdir(nvidia_path) if os.path.isdir(os.path.join(nvidia_path, d))]

    lib_paths = []
    for subdir in subdirs:
        lib_path = os.path.join(nvidia_path, subdir, 'lib')
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)

    if lib_paths:
        existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(lib_paths)
        if existing_ld_path:
            new_ld_path = f"{new_ld_path}:{existing_ld_path}"
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

# Now import the rest
from flax.training import train_state,checkpoints

import flax
import optax
import argparse
import os.path as osp

import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap,jit
import time
import sys

import gymnax
import gymnasium as gym
import stable_baselines3 as sb3

from src.control.mppi_class import MPPI
from src.control.mppi_class_optimized import create_optimized_generate_session
from src.utils.gym_jax import GymWrapper, EnvWithRender

# Import main's argument parsing
if __name__ == "__main__":
    # Read original main.py to get argument parsing
    import subprocess
    result = subprocess.run(['python', 'main.py', '--help'], capture_output=True, text=True)

    # For now, just hardcode the Walker2d experiment parameters
    class Args:
        gym_env = "Walker2d"
        num_traj = 500
        horizon = 50
        N_steps = 100
        N_steps_expert = 100
        rirl_iterations = 1
        reward_fn_updates = 15
        UB = True
        seed = 123
        lr = 1e-4
        lambda_ = 0.01
        Q = 1e-4
        P = 1e-2
        hidden_dim = 16
        save_images = False
        a_dim = None
        s_dim = None
        nu_dims = None
        frame_skip = 4
        dt = None
        N_iterations = 1000
        render_env = False

    args = Args()

    print("=== Walker2d RDIRL with Optimized MPPI ===")
    print(f"Environment: {args.gym_env}")
    print(f"RIRL Iterations: {args.rirl_iterations}")
    print(f"N_steps: {args.N_steps}")
    print(f"Horizon: {args.horizon}")
    print(f"Num Trajectories: {args.num_traj}")
    print("")

    # Create environment
    if args.gym_env.split("-")[0] in ["CartPole"]:
        env, env_params = gymnax.make(args.gym_env)
        env = GymWrapper(env)
        args.s_dim = env.observation_space.shape[0]
        args.a_dim = env.action_space.n
        args.nu_dims = env.action_space.n
        args.frame_skip = 1
        args.dt = 0.02
    else:
        env = gym.make(args.gym_env)
        if args.render_env:
            env = EnvWithRender(gym.make(args.gym_env, render_mode="human"))

        args.s_dim = env.observation_space.shape[0]
        args.a_dim = env.action_space.shape[0]
        args.nu_dims = args.a_dim
        args.dt = env.dt

    print(f"State dim: {args.s_dim}, Action dim: {args.a_dim}")
    print("")

    # Create MPPI instance
    policy = MPPI(
        args.s_dim,
        args.a_dim,
        args=args
    )

    # Create optimized version
    print("Creating optimized lax.scan-based session generator...")
    generate_session_optimized = create_optimized_generate_session(policy)
    print("Optimized generator created!")
    print("")

    # Initialize environment
    key = jax.random.PRNGKey(args.seed)
    state, info = env.reset(seed=args.seed)
    state = jnp.array(state, dtype=jnp.float32)

    # Initialize previous action sequence
    prev_action_seq = jnp.zeros((args.horizon, args.a_dim))

    # Dummy state_train for interface compatibility
    state_train = jnp.zeros((1, args.s_dim))

    print("Running optimized session generation...")
    print(f"This will execute {args.N_steps} timesteps using JIT-compiled lax.scan")
    print("")

    # Time the optimized lax.scan execution
    start_time = time.time()

    states, actions, total_reward, final_prev_action_seq = generate_session_optimized(
        state,
        key,
        prev_action_seq,
        state_train,
        args.N_steps,
        args.gym_env,
        args.frame_skip,
        args.dt,
        gail=False
    )

    # Block until computation completes
    states.block_until_ready()

    end_time = time.time()
    execution_time = end_time - start_time

    print("=== OPTIMIZED MPPI RESULTS ===")
    print(f"N_steps lax.scan outer loop execution time: {execution_time:.4f} seconds")
    print(f"Total reward: {float(total_reward):.4f}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print("")
    print("=== Experiment complete ===")

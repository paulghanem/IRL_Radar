# Simplified test script for optimized MPPI
import os
import sys

# Set JAX environment
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Configure CUDA paths
for path in sys.path:
    if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
        nvidia_path = os.path.join(path, 'nvidia')
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
        break

import jax
import jax.numpy as jnp
import gymnasium as gym
import time
from src.control.mppi_class import MPPI
from src.control.mppi_class_optimized import create_optimized_generate_session

print("=== Optimized MPPI Test for Walker2d ===")
print("")

# Setup args
class Args:
    gym_env = "Walker2d-v4"
    num_traj = 500
    horizon = 50
    N_steps = 100
    lambda_ = 0.01
    frame_skip = 4
    seed = 123

args = Args()

# Create environment
env = gym.make(args.gym_env)
args.s_dim = env.observation_space.shape[0]
args.a_dim = env.action_space.shape[0]
args.dt = env.dt

print(f"Environment: {args.gym_env}")
print(f"State dim: {args.s_dim}, Action dim: {args.a_dim}")
print(f"N_steps: {args.N_steps}, Horizon: {args.horizon}, Num trajectories: {args.num_traj}")
print("")

# Create MPPI controller
print("Creating MPPI controller...")
policy = MPPI(args.s_dim, args.a_dim, args=args)

# Create optimized generate_session function
print("Creating optimized lax.scan-based session generator...")
generate_session_optimized = create_optimized_generate_session(policy)
print("✓ Optimized generator created")
print("")

# Initialize
key = jax.random.PRNGKey(args.seed)
state, info = env.reset(seed=args.seed)
state = jnp.array(state, dtype=jnp.float32)
prev_action_seq = jnp.zeros((args.horizon, args.a_dim))
state_train = jnp.zeros((1, args.s_dim))

print("Running optimized session generation (lax.scan-based N_steps loop)...")
print("")

# Time the execution
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

print("=== RESULTS ===")
print(f"N_steps lax.scan outer loop execution time: {execution_time:.4f} seconds")
print(f"Total reward: {float(total_reward):.4f}")
print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print("")
print("✓ Experiment complete")

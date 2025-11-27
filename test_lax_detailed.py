"""
Detailed LAX test with timing and output analysis
"""
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

# Setup path
sys.path.insert(0, '.')

from src.utils.run_config import get_args
from src.control.PPO import PPOPolicy

print("=" * 70)
print("  LAX VERSION - Detailed Analysis")
print("=" * 70)

# Get arguments
args = get_args()
args.horizon = 50
args.N_steps = 100
args.gym_env = "Walker2d"
args.num_traj = 500
args.rirl_iterations = 1
args.UB = True
args.save_images = False

print(f"\nParameters:")
print(f"  N_steps: {args.N_steps}")
print(f"  horizon: {args.horizon}")
print(f"  num_traj: {args.num_traj}")
print(f"  Environment: {args.gym_env}")
print("=" * 70)
print()

# Initialize policy
print("Initializing policy...")
policy = PPOPolicy(args)

# Load demo data
import torch
D_demo = torch.load(f"expert_agents/{args.gym_env}/expert_trajs.pt")
D_demo = jnp.array(D_demo)
print(f"Loaded demo data: {D_demo.shape}")

# Dummy state_train
from flax.training import train_state
state_train = None  # Will use policy's internal state

print("\n" + "=" * 70)
print("Running LAX trajectory generation with detailed timing...")
print("=" * 70)

# Run once to JIT compile
print("\n[Warmup run for JIT compilation]")
start_warmup = time.time()
states, probs, actions, total_reward = policy.generate_session_lax(args, policy.state_train, D_demo)
warmup_time = time.time() - start_warmup
print(f"Warmup time (includes JIT compilation): {warmup_time:.4f} seconds")

# Convert back to arrays for analysis
states_arr = jnp.array(states)
rewards_per_step = []  # We'll need to recompute or extract

print("\n" + "=" * 70)
print("TRAJECTORY ANALYSIS")
print("=" * 70)

print(f"\nTotal trajectory length: {len(states)} steps")
print(f"Total reward: {total_reward:.4f}")
print(f"Average reward per step: {total_reward / len(states):.4f}")

# Analyze states
print(f"\nState shape per step: {states_arr.shape}")
print(f"\nFirst 5 states (positions, velocities):")
for i in range(min(5, len(states))):
    state = states_arr[i]
    print(f"  Step {i:3d}: {state[:6]}")  # Show first 6 elements

print(f"\nLast 5 states:")
for i in range(max(0, len(states)-5), len(states)):
    state = states_arr[i]
    print(f"  Step {i:3d}: {state[:6]}")

# Check for NaN or invalid values
if jnp.any(jnp.isnan(states_arr)):
    print("\n⚠️  WARNING: NaN detected in states!")
else:
    print("\n✓ No NaN values in states")

if jnp.any(jnp.isinf(states_arr)):
    print("⚠️  WARNING: Inf detected in states!")
else:
    print("✓ No Inf values in states")

# Analyze state statistics
print(f"\nState statistics across trajectory:")
print(f"  Mean: {jnp.mean(states_arr, axis=0)[:6]}")
print(f"  Std:  {jnp.std(states_arr, axis=0)[:6]}")
print(f"  Min:  {jnp.min(states_arr, axis=0)[:6]}")
print(f"  Max:  {jnp.max(states_arr, axis=0)[:6]}")

# Analyze actions
actions_arr = jnp.array(actions)
print(f"\nAction shape per step: {actions_arr.shape}")
print(f"Action statistics:")
print(f"  Mean: {jnp.mean(actions_arr, axis=0)}")
print(f"  Std:  {jnp.std(actions_arr, axis=0)}")
print(f"  Min:  {jnp.min(actions_arr, axis=0)}")
print(f"  Max:  {jnp.max(actions_arr, axis=0)}")

# Run timing test (after JIT)
print("\n" + "=" * 70)
print("TIMING TEST (5 runs after JIT)")
print("=" * 70)

times = []
for run in range(5):
    start = time.time()
    states, probs, actions, total_reward = policy.generate_session_lax(args, policy.state_train, D_demo)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {run+1}: {elapsed:.4f} seconds, reward: {total_reward:.4f}")

print(f"\nTiming statistics (after JIT):")
print(f"  Mean: {np.mean(times):.4f} seconds")
print(f"  Std:  {np.std(times):.4f} seconds")
print(f"  Min:  {np.min(times):.4f} seconds")
print(f"  Max:  {np.max(times):.4f} seconds")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)

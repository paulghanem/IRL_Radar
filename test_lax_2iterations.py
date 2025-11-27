"""
LAX test with 2 iterations - timing and output analysis for each iteration
"""
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

# Setup path
sys.path.insert(0, '.')

from src.control.PPO import PPOPolicy

print("=" * 70)
print("  LAX VERSION - 2 Iterations Detailed Analysis")
print("=" * 70)

# Simple args object
class Args:
    def __init__(self):
        self.horizon = 50
        self.N_steps = 100
        self.gym_env = "Walker2d"
        self.num_traj = 500
        self.rirl_iterations = 2
        self.UB = True
        self.save_images = False
        self.lr = 1e-4
        self.reward_fn_updates = 15
        self.lambda_ = 0.01
        self.num_samples = 500
        self.device = "cuda"
        self.method = "UB"
        self.s_dim = None
        self.a_dim = None

args = Args()

print(f"\nParameters:")
print(f"  N_steps: {args.N_steps}")
print(f"  horizon: {args.horizon}")
print(f"  num_traj: {args.num_traj}")
print(f"  rirl_iterations: {args.rirl_iterations}")
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

print("\n" + "=" * 70)
print("Running 2 RIRL iterations with detailed timing...")
print("=" * 70)

# Warmup run for JIT compilation
print("\n[Warmup run for JIT compilation]")
start_warmup = time.time()
states, probs, actions, total_reward = policy.generate_session_lax(args, policy.state_train, D_demo)
warmup_time = time.time() - start_warmup
print(f"Warmup time (includes JIT compilation): {warmup_time:.4f} seconds")

# Now run 2 iterations with timing
iteration_results = []

for iteration in range(2):
    print("\n" + "=" * 70)
    print(f"ITERATION {iteration + 1}")
    print("=" * 70)

    # Time this iteration
    start_iter = time.time()
    states, probs, actions, total_reward = policy.generate_session_lax(args, policy.state_train, D_demo)
    iter_time = time.time() - start_iter

    # Convert to arrays
    states_arr = jnp.array(states)
    actions_arr = jnp.array(actions)
    probs_arr = jnp.array(probs)

    # Store results
    result = {
        'iteration': iteration + 1,
        'time': iter_time,
        'states': states_arr,
        'actions': actions_arr,
        'probs': probs_arr,
        'total_reward': total_reward,
        'num_steps': len(states)
    }
    iteration_results.append(result)

    print(f"\nExecution time: {iter_time:.4f} seconds")
    print(f"Total steps: {len(states)}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward / len(states):.4f}")

    # Show first 5 states
    print(f"\nFirst 5 states (showing first 6 dimensions):")
    for i in range(min(5, len(states))):
        state = states_arr[i]
        print(f"  Step {i:3d}: {state[:6]}")

    # Show last 5 states
    print(f"\nLast 5 states:")
    for i in range(max(0, len(states)-5), len(states)):
        state = states_arr[i]
        print(f"  Step {i:3d}: {state[:6]}")

    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(states_arr))
    has_inf = jnp.any(jnp.isinf(states_arr))

    if has_nan:
        print("\n⚠️  WARNING: NaN detected in states!")
    else:
        print("\n✓ No NaN values in states")

    if has_inf:
        print("⚠️  WARNING: Inf detected in states!")
    else:
        print("✓ No Inf values in states")

    # State statistics
    print(f"\nState statistics (first 6 dimensions):")
    print(f"  Mean: {jnp.mean(states_arr, axis=0)[:6]}")
    print(f"  Std:  {jnp.std(states_arr, axis=0)[:6]}")
    print(f"  Min:  {jnp.min(states_arr, axis=0)[:6]}")
    print(f"  Max:  {jnp.max(states_arr, axis=0)[:6]}")

    # Action statistics
    print(f"\nAction statistics:")
    print(f"  Mean: {jnp.mean(actions_arr, axis=0)}")
    print(f"  Std:  {jnp.std(actions_arr, axis=0)}")
    print(f"  Min:  {jnp.min(actions_arr, axis=0)}")
    print(f"  Max:  {jnp.max(actions_arr, axis=0)}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Comparison Between Iterations")
print("=" * 70)

for i, result in enumerate(iteration_results):
    print(f"\nIteration {result['iteration']}:")
    print(f"  Execution time: {result['time']:.4f} seconds")
    print(f"  Total reward: {result['total_reward']:.4f}")
    print(f"  Steps: {result['num_steps']}")
    print(f"  Avg reward/step: {result['total_reward']/result['num_steps']:.4f}")

# Timing statistics
times = [r['time'] for r in iteration_results]
print(f"\nTiming statistics across iterations:")
print(f"  Mean: {np.mean(times):.4f} seconds")
print(f"  Std:  {np.std(times):.4f} seconds")
print(f"  Min:  {np.min(times):.4f} seconds")
print(f"  Max:  {np.max(times):.4f} seconds")

# Reward comparison
rewards = [r['total_reward'] for r in iteration_results]
print(f"\nReward statistics across iterations:")
print(f"  Mean: {np.mean(rewards):.4f}")
print(f"  Std:  {np.std(rewards):.4f}")
print(f"  Difference (Iter2 - Iter1): {rewards[1] - rewards[0]:.4f}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)

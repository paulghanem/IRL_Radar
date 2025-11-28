"""
Test Simplified Dynamics Accuracy

Compares SimplifiedWalker2d and SimplifiedHalfCheetah against real Gymnasium environments
to quantify approximation error and understand why GCL+MPPI may not converge well.

This helps answer:
1. How accurate are the simplified dynamics?
2. Do trajectories diverge over time?
3. Do rewards match well enough for IRL?
"""

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from src.control.simplified_walker import SimplifiedWalker, simplified_walker_step
from src.control.simplified_halfcheetah import SimplifiedHalfCheetah, simplified_halfcheetah_step

print("JAX devices:", jax.devices())
print("\n" + "="*70)
print("Testing Simplified Dynamics Accuracy vs Real Gymnasium Environments")
print("="*70)

# ============================================================
# Configuration
# ============================================================
ENV_NAME = "Walker2d-v4"  # Change to "HalfCheetah-v4" to test HalfCheetah
NUM_STEPS = 100  # Number of steps to test
NUM_TRIALS = 5  # Number of different action sequences to test
SEED = 42

# ============================================================
# Initialize Environments
# ============================================================
print(f"\nTesting: {ENV_NAME}")
print("-" * 70)

if ENV_NAME == "Walker2d-v4":
    simplified_env = SimplifiedWalker()
    s_dim = 18
    a_dim = 6
elif ENV_NAME == "HalfCheetah-v4":
    simplified_env = SimplifiedHalfCheetah()
    s_dim = 18
    a_dim = 6
else:
    raise ValueError(f"Unsupported environment: {ENV_NAME}")

# Create real Gymnasium environment
gym_env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)

print(f"  State dimension: {s_dim}")
print(f"  Action dimension: {a_dim}")
print(f"  Test steps: {NUM_STEPS}")
print(f"  Test trials: {NUM_TRIALS}")

# ============================================================
# Test Different Action Patterns
# ============================================================
action_patterns = [
    ("Zero Actions", lambda t: jnp.zeros(a_dim)),
    ("Constant Small", lambda t: jnp.ones(a_dim) * 0.1),
    ("Constant Medium", lambda t: jnp.ones(a_dim) * 0.5),
    ("Sine Wave", lambda t: jnp.sin(t * 0.1) * jnp.ones(a_dim)),
    ("Random", lambda t: jax.random.uniform(jax.random.PRNGKey(int(t)), (a_dim,), minval=-1.0, maxval=1.0))
]

results = {
    "pattern_names": [],
    "state_errors": [],
    "reward_errors": [],
    "state_errors_over_time": [],
    "reward_errors_over_time": []
}

for pattern_name, action_fn in action_patterns:
    print(f"\n{'='*70}")
    print(f"Testing Action Pattern: {pattern_name}")
    print('='*70)

    # Reset both environments
    gym_state, _ = gym_env.reset(seed=SEED)
    gym_state = np.array(gym_state)

    # Initialize simplified environment to match Gym
    simplified_state = jnp.array(gym_state)

    state_errors_trial = []
    reward_errors_trial = []

    for step in range(NUM_STEPS):
        # Generate action
        action = action_fn(step)
        action_np = np.array(action)

        # ========== Step Simplified Environment ==========
        simplified_next_state = simplified_env.step(simplified_state, action)
        simplified_reward = simplified_env.compute_reward(simplified_state, action, simplified_next_state)

        # ========== Step Real Gymnasium Environment ==========
        gym_next_state, gym_reward, terminated, truncated, info = gym_env.step(action_np)
        gym_next_state = np.array(gym_next_state)

        # ========== Compute Errors ==========
        state_error = np.linalg.norm(gym_next_state - np.array(simplified_next_state))
        reward_error = abs(gym_reward - float(simplified_reward))

        state_errors_trial.append(state_error)
        reward_errors_trial.append(reward_error)

        # Print periodic updates
        if step % 20 == 0:
            print(f"  Step {step:3d}: State Error = {state_error:.6f}, Reward Error = {reward_error:.6f}")

        # Update states
        simplified_state = simplified_next_state
        gym_state = gym_next_state

        # Break if environment terminates
        if terminated or truncated:
            print(f"  Environment terminated at step {step}")
            break

    # Store results
    results["pattern_names"].append(pattern_name)
    results["state_errors"].append(np.mean(state_errors_trial))
    results["reward_errors"].append(np.mean(reward_errors_trial))
    results["state_errors_over_time"].append(state_errors_trial)
    results["reward_errors_over_time"].append(reward_errors_trial)

    # Print summary
    print(f"\n  Summary for {pattern_name}:")
    print(f"    Average State Error:  {np.mean(state_errors_trial):.6f}")
    print(f"    Max State Error:      {np.max(state_errors_trial):.6f}")
    print(f"    Average Reward Error: {np.mean(reward_errors_trial):.6f}")
    print(f"    Max Reward Error:     {np.max(reward_errors_trial):.6f}")

# ============================================================
# Overall Summary
# ============================================================
print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

for i, pattern_name in enumerate(results["pattern_names"]):
    print(f"\n{pattern_name}:")
    print(f"  Avg State Error:  {results['state_errors'][i]:.6f}")
    print(f"  Avg Reward Error: {results['reward_errors'][i]:.6f}")

print(f"\n{'='*70}")
print(f"Overall Average State Error:  {np.mean(results['state_errors']):.6f}")
print(f"Overall Average Reward Error: {np.mean(results['reward_errors']):.6f}")
print(f"{'='*70}")

# ============================================================
# Visualization
# ============================================================
print("\n\nGenerating plots...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: State Error Over Time
ax = axes[0]
for i, pattern_name in enumerate(results["pattern_names"]):
    ax.plot(results["state_errors_over_time"][i], label=pattern_name, linewidth=2)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("State Error (L2 Norm)", fontsize=12)
ax.set_title(f"{ENV_NAME}: State Error Over Time", fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Reward Error Over Time
ax = axes[1]
for i, pattern_name in enumerate(results["pattern_names"]):
    ax.plot(results["reward_errors_over_time"][i], label=pattern_name, linewidth=2)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Reward Error (Absolute)", fontsize=12)
ax.set_title(f"{ENV_NAME}: Reward Error Over Time", fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"{ENV_NAME.replace('-', '_')}_dynamics_accuracy.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved: {plot_filename}")

# ============================================================
# Save Results
# ============================================================
results_filename = f"{ENV_NAME.replace('-', '_')}_dynamics_accuracy.npz"
# Convert lists of varying lengths to object arrays
np.savez(
    results_filename,
    pattern_names=np.array(results["pattern_names"]),
    state_errors=np.array(results["state_errors"]),
    reward_errors=np.array(results["reward_errors"]),
    state_errors_over_time=np.array(results["state_errors_over_time"], dtype=object),
    reward_errors_over_time=np.array(results["reward_errors_over_time"], dtype=object)
)
print(f"Results saved: {results_filename}")

# ============================================================
# Analysis & Recommendations
# ============================================================
print("\n" + "="*70)
print("ANALYSIS & RECOMMENDATIONS")
print("="*70)

avg_state_error = np.mean(results['state_errors'])
avg_reward_error = np.mean(results['reward_errors'])

print(f"\n1. Dynamics Accuracy:")
if avg_state_error < 0.01:
    print(f"   ✅ EXCELLENT: State error = {avg_state_error:.6f}")
    print(f"      Simplified dynamics closely match real environment.")
elif avg_state_error < 0.1:
    print(f"   ⚠️  MODERATE: State error = {avg_state_error:.6f}")
    print(f"      Simplified dynamics are reasonable but not perfect.")
else:
    print(f"   ❌ POOR: State error = {avg_state_error:.6f}")
    print(f"      Simplified dynamics diverge significantly from real environment.")
    print(f"      This explains poor GCL+MPPI convergence!")

print(f"\n2. Reward Accuracy:")
if avg_reward_error < 0.01:
    print(f"   ✅ EXCELLENT: Reward error = {avg_reward_error:.6f}")
    print(f"      Reward function matches well.")
elif avg_reward_error < 0.1:
    print(f"   ⚠️  MODERATE: Reward error = {avg_reward_error:.6f}")
    print(f"      Reward function has some discrepancies.")
else:
    print(f"   ❌ POOR: Reward error = {avg_reward_error:.6f}")
    print(f"      Reward function differs significantly!")

print(f"\n3. Implications for GCL+MPPI:")
if avg_state_error > 0.05 or avg_reward_error > 0.05:
    print(f"   ⚠️  Expert demonstrations from real {ENV_NAME} may not match")
    print(f"      simplified dynamics well. Consider:")
    print(f"      a) Use expert demos from simplified environment")
    print(f"      b) Improve simplified dynamics approximation")
    print(f"      c) Use real MuJoCo with MJX for both expert & agent")
else:
    print(f"   ✅ Dynamics are accurate enough for IRL.")
    print(f"      Convergence issues likely due to other factors:")
    print(f"      - MPPI parameters (horizon, lambda, num_samples)")
    print(f"      - Cost learning rate too high")
    print(f"      - Need RGCL for stability")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)

gym_env.close()

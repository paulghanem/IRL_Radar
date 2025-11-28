"""
Compare Expert Rewards: Real Walker2d vs SimplifiedWalker2d

Generates expert trajectories using PPO policy (like main.py does),
then compares:
1. Rewards from real Walker2d-v4 environment
2. Rewards computed by SimplifiedWalker2d using same states/actions

This quantifies how well SimplifiedWalker2d can replicate expert behavior rewards.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import stats

from utils.helpers import GenerateDemo
from src.control.simplified_walker import SimplifiedWalker

print("=" * 70)
print("Expert Reward Comparison: Real Walker2d vs SimplifiedWalker2d")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================
ENV_NAME = "Walker2d"  # This matches the expert_agents folder structure
MAX_FRAMES = 200
SEED = 42

# ============================================================
# Generate Expert Trajectories
# ============================================================
print(f"\nGenerating expert trajectories using PPO policy...")
print(f"Environment: {ENV_NAME}")
print(f"Max frames: {MAX_FRAMES}")
print(f"Seed: {SEED}")
print()

demo_generator = GenerateDemo(ENV_NAME, max_frames=MAX_FRAMES)
states, actions, cumulative_rewards, vec_env = demo_generator.generate_demo(seed=SEED)

# Get per-step rewards from cumulative
real_rewards = np.diff(cumulative_rewards, prepend=0.0)

print(f"\nExpert trajectory generated:")
print(f"  Number of steps: {len(states)}")
print(f"  Total return: {np.sum(real_rewards):.2f}")
print(f"  Mean reward per step: {np.mean(real_rewards):.4f}")

# ============================================================
# Compute SimplifiedWalker2d Rewards
# ============================================================
print(f"\nComputing SimplifiedWalker2d rewards for same trajectory...")

simplified_env = SimplifiedWalker()
simplified_rewards = []

for i in range(len(states) - 1):
    # Current state, action, next state
    state = jnp.array(states[i])
    action = jnp.array(actions[i])
    next_state = jnp.array(states[i + 1])

    # Compute reward using SimplifiedWalker2d
    simplified_reward = simplified_env.compute_reward(state, action, next_state)
    simplified_rewards.append(float(simplified_reward))

simplified_rewards = np.array(simplified_rewards)

# Match dimensions (real_rewards has one more element than simplified_rewards)
real_rewards = real_rewards[:len(simplified_rewards)]

print(f"\nSimplifiedWalker2d rewards computed:")
print(f"  Number of steps: {len(simplified_rewards)}")
print(f"  Total return: {np.sum(simplified_rewards):.2f}")
print(f"  Mean reward per step: {np.mean(simplified_rewards):.4f}")

# ============================================================
# Compute Statistics
# ============================================================
print("\n" + "=" * 70)
print("REWARD COMPARISON STATISTICS")
print("=" * 70)

# Absolute errors
reward_errors = np.abs(real_rewards - simplified_rewards)
relative_errors = np.abs(real_rewards - simplified_rewards) / (np.abs(real_rewards) + 1e-8)

print(f"\nAbsolute Reward Error:")
print(f"  Mean:   {np.mean(reward_errors):.4f}")
print(f"  Median: {np.median(reward_errors):.4f}")
print(f"  Std:    {np.std(reward_errors):.4f}")
print(f"  Max:    {np.max(reward_errors):.4f}")
print(f"  95th percentile: {np.percentile(reward_errors, 95):.4f}")

print(f"\nRelative Reward Error (%):")
print(f"  Mean:   {np.mean(relative_errors) * 100:.2f}%")
print(f"  Median: {np.median(relative_errors) * 100:.2f}%")

# Correlation
corr, p_value = stats.pearsonr(real_rewards, simplified_rewards)
print(f"\nReward Correlation:")
print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")

# Total return comparison
real_return = np.sum(real_rewards)
simplified_return = np.sum(simplified_rewards)
return_error = np.abs(real_return - simplified_return)
return_error_pct = (return_error / np.abs(real_return)) * 100

print(f"\nTotal Return Comparison:")
print(f"  Real Walker2d:       {real_return:.2f}")
print(f"  SimplifiedWalker2d:  {simplified_return:.2f}")
print(f"  Absolute difference: {return_error:.2f}")
print(f"  Relative difference: {return_error_pct:.2f}%")

# ============================================================
# Breakdown by Reward Sign
# ============================================================
print(f"\n{'=' * 70}")
print("Breakdown by Reward Sign")
print('=' * 70)

positive_mask = real_rewards > 0
negative_mask = real_rewards <= 0

if np.sum(positive_mask) > 0:
    print(f"\nPositive Rewards (n={np.sum(positive_mask)}):")
    print(f"  Real mean:       {np.mean(real_rewards[positive_mask]):.4f}")
    print(f"  Simplified mean: {np.mean(simplified_rewards[positive_mask]):.4f}")
    print(f"  Error:           {np.mean(reward_errors[positive_mask]):.4f}")

if np.sum(negative_mask) > 0:
    print(f"\nNegative/Zero Rewards (n={np.sum(negative_mask)}):")
    print(f"  Real mean:       {np.mean(real_rewards[negative_mask]):.4f}")
    print(f"  Simplified mean: {np.mean(simplified_rewards[negative_mask]):.4f}")
    print(f"  Error:           {np.mean(reward_errors[negative_mask]):.4f}")

# ============================================================
# Visualization
# ============================================================
print("\n\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reward trajectories over time
ax = axes[0, 0]
ax.plot(real_rewards, label='Real Walker2d', linewidth=2, alpha=0.8)
ax.plot(simplified_rewards, label='SimplifiedWalker2d', linewidth=2, alpha=0.8, linestyle='--')
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Expert Trajectory Rewards', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reward correlation scatter
ax = axes[0, 1]
ax.scatter(real_rewards, simplified_rewards, alpha=0.6, s=30)
ax.plot([real_rewards.min(), real_rewards.max()],
        [real_rewards.min(), real_rewards.max()],
        'r--', linewidth=2, label='Perfect match')
ax.set_xlabel('Real Walker2d Reward', fontsize=11)
ax.set_ylabel('SimplifiedWalker2d Reward', fontsize=11)
ax.set_title(f'Reward Correlation (r={corr:.3f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Absolute error over time
ax = axes[1, 0]
ax.plot(reward_errors, linewidth=2, color='red', alpha=0.7)
ax.axhline(np.mean(reward_errors), color='black', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(reward_errors):.3f}')
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Absolute Reward Error', fontsize=11)
ax.set_title('Reward Error Over Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax = axes[1, 1]
ax.hist(reward_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.mean(reward_errors), color='r', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(reward_errors):.3f}')
ax.axvline(np.median(reward_errors), color='g', linestyle='--',
           linewidth=2, label=f'Median: {np.median(reward_errors):.3f}')
ax.set_xlabel('Absolute Reward Error', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Reward Error Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = "expert_reward_comparison.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved: {plot_filename}")

# ============================================================
# Save Results
# ============================================================
results_filename = "expert_reward_comparison.npz"
np.savez(
    results_filename,
    states=states[:len(simplified_rewards)+1],  # +1 because states has one more element
    actions=actions[:len(simplified_rewards)],
    real_rewards=real_rewards,
    simplified_rewards=simplified_rewards,
    reward_errors=reward_errors
)
print(f"Results saved: {results_filename}")

# ============================================================
# Analysis
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

mean_error = np.mean(reward_errors)
mean_rel_error = np.mean(relative_errors) * 100

print(f"\n1. Reward Matching Quality:")
if mean_error < 0.05:
    print(f"   EXCELLENT: Mean reward error = {mean_error:.4f}")
    print(f"   SimplifiedWalker2d rewards closely match real environment!")
elif mean_error < 0.2:
    print(f"   GOOD: Mean reward error = {mean_error:.4f}")
    print(f"   SimplifiedWalker2d rewards are reasonably accurate.")
elif mean_error < 0.5:
    print(f"   MODERATE: Mean reward error = {mean_error:.4f}")
    print(f"   SimplifiedWalker2d rewards have noticeable discrepancies.")
else:
    print(f"   POOR: Mean reward error = {mean_error:.4f}")
    print(f"   SimplifiedWalker2d rewards differ significantly from real environment.")

print(f"\n2. Reward Correlation:")
if corr > 0.95:
    print(f"   EXCELLENT: r={corr:.4f}")
    print(f"   Rewards are highly correlated - trends match well.")
elif corr > 0.8:
    print(f"   GOOD: r={corr:.4f}")
    print(f"   Rewards are well correlated - general trends match.")
elif corr > 0.5:
    print(f"   MODERATE: r={corr:.4f}")
    print(f"   Some correlation but with noticeable differences.")
else:
    print(f"   POOR: r={corr:.4f}")
    print(f"   Weak correlation - rewards follow different patterns.")

print(f"\n3. Total Return Error:")
if return_error_pct < 5:
    print(f"   EXCELLENT: {return_error_pct:.2f}% difference")
    print(f"   Total returns match very well.")
elif return_error_pct < 15:
    print(f"   GOOD: {return_error_pct:.2f}% difference")
    print(f"   Total returns are reasonably close.")
elif return_error_pct < 30:
    print(f"   MODERATE: {return_error_pct:.2f}% difference")
    print(f"   Total returns have noticeable gap.")
else:
    print(f"   POOR: {return_error_pct:.2f}% difference")
    print(f"   Total returns differ significantly.")

print(f"\n4. Implications for GCL+MPPI:")
if mean_error < 0.2 and corr > 0.8:
    print(f"   SimplifiedWalker2d rewards match expert behavior well enough.")
    print(f"   Reward mismatch is NOT the primary issue for GCL convergence.")
    print(f"   Focus on:")
    print(f"     - MPPI hyperparameters (horizon, lambda, num_samples)")
    print(f"     - Cost learning rate and stability (try RGCL)")
    print(f"     - State dynamics mismatch (see single-step test)")
else:
    print(f"   SimplifiedWalker2d rewards don't match expert well.")
    print(f"   This explains GCL convergence issues!")
    print(f"   Recommendations:")
    print(f"     - Generate expert demos FROM SimplifiedWalker2d (not real Walker2d)")
    print(f"     - OR switch to MJX for accurate physics")
    print(f"     - Reward error: {mean_error:.4f}, Correlation: {corr:.4f}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)

vec_env.close()

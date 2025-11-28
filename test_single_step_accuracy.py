"""
Test Single-Step Dynamics Accuracy

Tests the simplified dynamics for ONE time step across many different control inputs.
This isolates the single-step prediction error from accumulated trajectory error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy import stats

from src.control.simplified_walker import SimplifiedWalker

print("JAX devices:", jax.devices())
print("\n" + "="*70)
print("Single-Step Dynamics Accuracy Test")
print("="*70)

# ============================================================
# Configuration
# ============================================================
ENV_NAME = "Walker2d-v4"
NUM_ACTIONS = 1000  # Test 1000 different actions
SEED = 42

# ============================================================
# Initialize Environments
# ============================================================
print(f"\nEnvironment: {ENV_NAME}")
print(f"Number of test actions: {NUM_ACTIONS}")

simplified_env = SimplifiedWalker()
gym_env = gym.make(ENV_NAME, exclude_current_positions_from_observation=False)

s_dim = 18
a_dim = 6

# ============================================================
# Generate Test Actions
# ============================================================
print("\nGenerating test actions...")

# Create diverse action set
rng = np.random.RandomState(SEED)

# Different action types
actions = []

# 1. Zero action
actions.append(np.zeros(a_dim))

# 2. Unit actions (each dimension individually)
for i in range(a_dim):
    action = np.zeros(a_dim)
    action[i] = 1.0
    actions.append(action)
    action = np.zeros(a_dim)
    action[i] = -1.0
    actions.append(action)

# 3. Small random actions (-0.3 to 0.3)
for _ in range(200):
    actions.append(rng.uniform(-0.3, 0.3, a_dim))

# 4. Medium random actions (-0.7 to 0.7)
for _ in range(200):
    actions.append(rng.uniform(-0.7, 0.7, a_dim))

# 5. Large random actions (-1.0 to 1.0)
for _ in range(NUM_ACTIONS - len(actions)):
    actions.append(rng.uniform(-1.0, 1.0, a_dim))

actions = np.array(actions)
print(f"Generated {len(actions)} test actions")

# ============================================================
# Test Single-Step Predictions
# ============================================================
print("\nTesting single-step predictions...")

state_errors = []
reward_errors = []
action_magnitudes = []

# Get initial state
gym_state, _ = gym_env.reset(seed=SEED)
gym_state = np.array(gym_state)
initial_state = jnp.array(gym_state)

for i, action in enumerate(actions):
    # Reset to same initial state for each test
    gym_env.reset(seed=SEED)

    # ========== Simplified Environment ==========
    simplified_next_state = simplified_env.step(initial_state, jnp.array(action))
    simplified_reward = simplified_env.compute_reward(
        initial_state, jnp.array(action), simplified_next_state
    )

    # ========== Real Gymnasium Environment ==========
    gym_env.reset(seed=SEED)  # Reset to initial state
    gym_next_state, gym_reward, _, _, _ = gym_env.step(action)
    gym_next_state = np.array(gym_next_state)

    # ========== Compute Errors ==========
    state_error = np.linalg.norm(gym_next_state - np.array(simplified_next_state))
    reward_error = abs(gym_reward - float(simplified_reward))
    action_magnitude = np.linalg.norm(action)

    state_errors.append(state_error)
    reward_errors.append(reward_error)
    action_magnitudes.append(action_magnitude)

    if i % 200 == 0:
        print(f"  Tested {i}/{len(actions)} actions...")

state_errors = np.array(state_errors)
reward_errors = np.array(reward_errors)
action_magnitudes = np.array(action_magnitudes)

# ============================================================
# Statistics
# ============================================================
print("\n" + "="*70)
print("SINGLE-STEP ACCURACY STATISTICS")
print("="*70)

print(f"\nState Error:")
print(f"  Mean:   {np.mean(state_errors):.6f}")
print(f"  Median: {np.median(state_errors):.6f}")
print(f"  Std:    {np.std(state_errors):.6f}")
print(f"  Min:    {np.min(state_errors):.6f}")
print(f"  Max:    {np.max(state_errors):.6f}")
print(f"  95th percentile: {np.percentile(state_errors, 95):.6f}")

print(f"\nReward Error:")
print(f"  Mean:   {np.mean(reward_errors):.6f}")
print(f"  Median: {np.median(reward_errors):.6f}")
print(f"  Std:    {np.std(reward_errors):.6f}")
print(f"  Min:    {np.min(reward_errors):.6f}")
print(f"  Max:    {np.max(reward_errors):.6f}")
print(f"  95th percentile: {np.percentile(reward_errors, 95):.6f}")

# Correlation with action magnitude
corr_state, p_state = stats.pearsonr(action_magnitudes, state_errors)
corr_reward, p_reward = stats.pearsonr(action_magnitudes, reward_errors)

print(f"\nCorrelation with Action Magnitude:")
print(f"  State Error:  r={corr_state:.3f} (p={p_state:.4f})")
print(f"  Reward Error: r={corr_reward:.3f} (p={p_reward:.4f})")

# ============================================================
# Breakdown by Action Magnitude
# ============================================================
print(f"\n{'='*70}")
print("Error by Action Magnitude")
print('='*70)

bins = [(0.0, 0.3, "Small"), (0.3, 0.7, "Medium"), (0.7, 1.5, "Large")]

for min_mag, max_mag, label in bins:
    mask = (action_magnitudes >= min_mag) & (action_magnitudes < max_mag)
    if np.sum(mask) > 0:
        print(f"\n{label} Actions (||a|| in [{min_mag:.1f}, {max_mag:.1f}]):")
        print(f"  Count: {np.sum(mask)}")
        print(f"  State Error:  {np.mean(state_errors[mask]):.6f} ± {np.std(state_errors[mask]):.6f}")
        print(f"  Reward Error: {np.mean(reward_errors[mask]):.6f} ± {np.std(reward_errors[mask]):.6f}")

# ============================================================
# Visualizations
# ============================================================
print("\n\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: State Error Distribution
ax = axes[0, 0]
ax.hist(state_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(state_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(state_errors):.3f}')
ax.axvline(np.median(state_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(state_errors):.3f}')
ax.set_xlabel('State Error (L2 Norm)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('State Error Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reward Error Distribution
ax = axes[0, 1]
ax.hist(reward_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.mean(reward_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(reward_errors):.3f}')
ax.axvline(np.median(reward_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(reward_errors):.3f}')
ax.set_xlabel('Reward Error (Absolute)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Reward Error Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: State Error vs Action Magnitude
ax = axes[1, 0]
scatter = ax.scatter(action_magnitudes, state_errors, alpha=0.5, s=10)
ax.set_xlabel('Action Magnitude (L2 Norm)', fontsize=11)
ax.set_ylabel('State Error', fontsize=11)
ax.set_title(f'State Error vs Action Magnitude (r={corr_state:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(action_magnitudes, state_errors, 1)
p = np.poly1d(z)
ax.plot(action_magnitudes, p(action_magnitudes), "r--", linewidth=2, label='Linear fit')
ax.legend()

# Plot 4: Reward Error vs Action Magnitude
ax = axes[1, 1]
scatter = ax.scatter(action_magnitudes, reward_errors, alpha=0.5, s=10, color='orange')
ax.set_xlabel('Action Magnitude (L2 Norm)', fontsize=11)
ax.set_ylabel('Reward Error', fontsize=11)
ax.set_title(f'Reward Error vs Action Magnitude (r={corr_reward:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(action_magnitudes, reward_errors, 1)
p = np.poly1d(z)
ax.plot(action_magnitudes, p(action_magnitudes), "r--", linewidth=2, label='Linear fit')
ax.legend()

plt.tight_layout()
plot_filename = "single_step_accuracy.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved: {plot_filename}")

# ============================================================
# Save Results
# ============================================================
results_filename = "single_step_accuracy.npz"
np.savez(
    results_filename,
    state_errors=state_errors,
    reward_errors=reward_errors,
    action_magnitudes=action_magnitudes,
    actions=actions
)
print(f"Results saved: {results_filename}")

# ============================================================
# Analysis
# ============================================================
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

mean_state_error = np.mean(state_errors)
mean_reward_error = np.mean(reward_errors)

print(f"\n1. Single-Step Accuracy:")
if mean_state_error < 0.1:
    print(f"   EXCELLENT: Mean state error = {mean_state_error:.6f}")
    print(f"   Single-step predictions are very accurate!")
elif mean_state_error < 0.5:
    print(f"   GOOD: Mean state error = {mean_state_error:.6f}")
    print(f"   Single-step predictions are reasonably accurate.")
elif mean_state_error < 2.0:
    print(f"   MODERATE: Mean state error = {mean_state_error:.6f}")
    print(f"   Single-step predictions have moderate errors.")
else:
    print(f"   POOR: Mean state error = {mean_state_error:.6f}")
    print(f"   Single-step predictions have large errors.")

print(f"\n2. Error vs Action Magnitude:")
if corr_state > 0.5:
    print(f"   Strong positive correlation (r={corr_state:.3f})")
    print(f"   Larger actions → larger errors")
    print(f"   Model struggles with large control inputs")
elif corr_state > 0.2:
    print(f"   Moderate positive correlation (r={corr_state:.3f})")
    print(f"   Some dependency on action magnitude")
else:
    print(f"   Weak correlation (r={corr_state:.3f})")
    print(f"   Error is relatively independent of action magnitude")

print(f"\n3. Implications for Multi-Step Trajectories:")
if mean_state_error < 0.5:
    print(f"   Single-step error is small.")
    print(f"   Multi-step error likely from accumulation.")
    print(f"   Better integration or longer frame_skip might help.")
else:
    print(f"   Single-step error is already significant.")
    print(f"   Fundamental dynamics approximation needs improvement.")
    print(f"   Consider using real MuJoCo physics (MJX).")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)

gym_env.close()

"""
Plot Simplified Walker2d GCL + MPPI Training Results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load results
rewards = np.load('simplified_walker_rewards.npy')
losses = np.load('simplified_walker_losses.npy')

# Expert reward
expert_reward = 100.14

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Agent Rewards
iterations = np.arange(1, len(rewards) + 1)
ax1.plot(iterations, rewards, 'b-', linewidth=2, label='Agent Reward', marker='o', markersize=4)

# Add expert baseline
ax1.axhline(y=expert_reward, color='r', linestyle='--', linewidth=2, label=f'Expert ({expert_reward:.2f})')

# Add moving average
if len(rewards) >= 5:
    window = 5
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(iterations[window-1:], moving_avg, 'g-', linewidth=2, alpha=0.7, label='5-iter Moving Avg')

# Annotations
peak_idx = np.argmax(rewards)
peak_reward = rewards[peak_idx]
ax1.annotate(f'Peak: {peak_reward:.2f}',
             xy=(iterations[peak_idx], peak_reward),
             xytext=(iterations[peak_idx]+2, peak_reward+2),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

final_reward = rewards[-1]
ax1.annotate(f'Final: {final_reward:.2f}',
             xy=(iterations[-1], final_reward),
             xytext=(iterations[-1]-5, final_reward-3),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
             fontsize=10, color='blue', fontweight='bold')

ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Reward', fontsize=12)
ax1.set_title('Simplified Walker2d - GCL + MPPI Training Performance', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(rewards) + 1)

# Plot 2: Cost Function Losses
valid_losses = losses[~np.isnan(losses)]
valid_iterations = iterations[~np.isnan(losses)]

ax2.plot(valid_iterations, valid_losses, 'purple', linewidth=2, marker='s', markersize=4, label='Cost Loss')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Cost Function Training Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, len(rewards) + 1)

# Add text box with summary stats
summary_text = f"""Summary Statistics:
Initial Reward: {rewards[0]:.2f}
Final Reward: {rewards[-1]:.2f}
Peak Reward: {np.max(rewards):.2f}
Average Reward: {np.mean(rewards):.2f}
Expert Reward: {expert_reward:.2f}
Achievement: {(rewards[-1]/expert_reward)*100:.1f}%

Cost Losses:
Valid: {len(valid_losses)}/{len(losses)}
NaN: {np.sum(np.isnan(losses))}/{len(losses)}
Avg Loss: {np.mean(valid_losses):.4f}"""

ax2.text(0.98, 0.02, summary_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('simplified_walker_training_results.png', dpi=300, bbox_inches='tight')
print("Plot saved: simplified_walker_training_results.png")

# Print summary
print("\n" + "="*60)
print("Simplified Walker2d GCL + MPPI Training Summary")
print("="*60)
print(f"Initial Reward:  {rewards[0]:.2f}")
print(f"Final Reward:    {rewards[-1]:.2f}")
print(f"Peak Reward:     {np.max(rewards):.2f} (iteration {np.argmax(rewards)+1})")
print(f"Average Reward:  {np.mean(rewards):.2f}")
print(f"Std Dev:         {np.std(rewards):.2f}")
print(f"\nExpert Reward:   {expert_reward:.2f}")
print(f"Achievement:     {(rewards[-1]/expert_reward)*100:.1f}% of expert")
print(f"\nCost Function:")
print(f"  Valid Losses:  {len(valid_losses)}/{len(losses)} ({100*len(valid_losses)/len(losses):.1f}%)")
print(f"  NaN Losses:    {np.sum(np.isnan(losses))}/{len(losses)} ({100*np.sum(np.isnan(losses))/len(losses):.1f}%)")
print(f"  Average Loss:  {np.mean(valid_losses):.4f}")
print("="*60)

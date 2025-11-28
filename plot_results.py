"""
Plot training results from CartPole PPO-IRL experiment
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# Load results
results_dir = "results/CartPole-v1/gcl-ppo"
cost_file = os.path.join(results_dir, "cost_seed=42.npy")
expert_file = os.path.join(results_dir, "expert_cost_seed=42.npy")
loss_file = os.path.join(results_dir, "loss_seed=42.npy")

if os.path.exists(cost_file):
    agent_rewards = np.load(cost_file)[0]  # [0] to get first run
    expert_rewards = np.load(expert_file)[0]
    losses = np.load(loss_file)

    iterations = np.arange(1, len(agent_rewards) + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Agent vs Expert Rewards
    ax1 = axes[0]
    ax1.plot(iterations, agent_rewards, 'b-', linewidth=2, label='Agent Reward', marker='o', markersize=4)
    ax1.axhline(y=expert_rewards[0], color='r', linestyle='--', linewidth=2, label=f'Expert Reward ({expert_rewards[0]:.0f})')

    # Add moving average
    window = 10
    if len(agent_rewards) >= window:
        moving_avg = np.convolve(agent_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(iterations[window-1:], moving_avg, 'g-', linewidth=2, alpha=0.7, label=f'{window}-iteration Moving Avg')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('PPO-IRL Training Progress on CartPole-v1', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add milestone annotations
    ax1.annotate(f'Start: {agent_rewards[0]:.0f}',
                xy=(1, agent_rewards[0]), xytext=(5, agent_rewards[0]+10),
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    peak_idx = np.argmax(agent_rewards)
    ax1.annotate(f'Peak: {agent_rewards[peak_idx]:.0f}',
                xy=(peak_idx+1, agent_rewards[peak_idx]), xytext=(peak_idx+5, agent_rewards[peak_idx]+5),
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.annotate(f'Final: {agent_rewards[-1]:.0f}',
                xy=(len(agent_rewards), agent_rewards[-1]), xytext=(len(agent_rewards)-10, agent_rewards[-1]+8),
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7))

    # Plot 2: Cost Function Loss
    ax2 = axes[1]

    # Filter out NaN values for plotting
    valid_losses = [(i+1, loss) for i, loss in enumerate(losses) if not np.isnan(loss)]
    if valid_losses:
        valid_iters, valid_loss_vals = zip(*valid_losses)
        ax2.plot(valid_iters, valid_loss_vals, 'ro-', linewidth=2, markersize=6, label='Valid Loss')

    # Show NaN occurrences
    nan_iters = [i+1 for i, loss in enumerate(losses) if np.isnan(loss)]
    if nan_iters:
        ax2.scatter(nan_iters, [0]*len(nan_iters), color='gray', marker='x', s=50,
                   alpha=0.5, label=f'NaN Loss ({len(nan_iters)} occurrences)')

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost Function Loss', fontsize=12)
    ax2.set_title('Cost Function Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('cartpole_ppo_irl_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Plot saved as 'cartpole_ppo_irl_results.png'")

    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total Iterations: {len(agent_rewards)}")
    print(f"Initial Reward: {agent_rewards[0]:.2f}")
    print(f"Final Reward: {agent_rewards[-1]:.2f}")
    print(f"Peak Reward: {agent_rewards[peak_idx]:.2f} (iteration {peak_idx+1})")
    print(f"Average Reward: {np.mean(agent_rewards):.2f}")
    print(f"Std Dev: {np.std(agent_rewards):.2f}")
    print(f"\nExpert Reward: {expert_rewards[0]:.2f}")
    print(f"Performance Gap: {expert_rewards[0] - agent_rewards[-1]:.2f}")
    print(f"Achievement: {(agent_rewards[-1]/expert_rewards[0])*100:.2f}% of expert")

    print(f"\n{'='*60}")
    print("LOSS STATISTICS")
    print("="*60)
    print(f"Valid Losses: {len(valid_losses)}/{len(losses)} ({(len(valid_losses)/len(losses)*100):.1f}%)")
    print(f"NaN Losses: {len(nan_iters)}/{len(losses)} ({(len(nan_iters)/len(losses)*100):.1f}%)")
    if valid_losses:
        valid_loss_vals = np.array(valid_loss_vals)
        print(f"Loss Range: [{np.min(valid_loss_vals):.3f}, {np.max(valid_loss_vals):.3f}]")
        print(f"Average Loss: {np.mean(valid_loss_vals):.3f}")

else:
    print(f"Results not found at: {results_dir}")
    print("Please run training first: python main_ppo.py")

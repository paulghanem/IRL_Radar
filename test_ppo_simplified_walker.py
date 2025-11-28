"""
Unit test for PPO on SimplifiedWalker2d

This test verifies that PPO can learn a good policy on SimplifiedWalker2d
before integrating it with GCL in main_simplified.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.control.simplified_walker import SimplifiedWalker
from src.control.PPO_walker import PPOWalker


class Args:
    """Configuration for PPO testing"""
    def __init__(self):
        self.seed = 42
        self.s_dim = 18  # Walker2d state dimension
        self.a_dim = 6   # Walker2d action dimension
        self.N_steps = 500  # Steps per episode
        self.gym_env = "SimplifiedWalker2d"


def test_ppo_simplified_walker():
    """Test PPO learning on SimplifiedWalker2d"""
    print("="*70)
    print("PPO SimplifiedWalker2d Unit Test")
    print("="*70)

    args = Args()

    # Initialize SimplifiedWalker2d
    print("\n1. Initializing SimplifiedWalker2d...")
    walker = SimplifiedWalker(dt=0.002, frame_skip=4)
    print(f"   State dim: {walker.state_dim}")
    print(f"   Action dim: {walker.action_dim}")
    print(f"   Effective dt: {walker.dt * walker.frame_skip}")

    # Initialize PPO agent
    print("\n2. Initializing PPO agent...")
    ppo_agent = PPOWalker(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        walker=walker,
        args=args,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=1e-3,
        rollout_length=args.N_steps,
        buffer_mix=10
    )
    print(f"   Actor network: {args.s_dim} -> 256 -> 256 -> {args.a_dim}")
    print(f"   Critic network: {args.s_dim} -> 256 -> 256 -> 1")

    # Create dummy expert demonstrations (just for interface compatibility)
    print("\n3. Creating dummy demo data...")
    init_state = jnp.array([
        0.0,    # x position
        1.25,   # z height (standing)
        0.0,    # root angle
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # joint angles
        0.0,    # x velocity
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # velocities
    ])

    D_demo = jnp.tile(init_state, (args.N_steps, 1))
    D_demo = jnp.concatenate([
        D_demo,
        jnp.ones((args.N_steps, 1)),  # Probs
        jnp.zeros((args.N_steps, args.a_dim))  # Dummy actions
    ], axis=1)

    # Training loop
    print("\n4. Training PPO...")
    print("   Episodes: 100")
    print("   Steps per episode: 500")
    print("   Updates per episode: 5")
    print()

    num_episodes = 100
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        # Generate rollout
        states, probs, actions, total_reward = ppo_agent.generate_session_lax(
            args=args,
            state_train=None,  # Not using GCL yet
            D_demo=D_demo,
            iteration=episode  # Vary seed per episode
        )

        episode_rewards.append(total_reward)
        episode_lengths.append(len(states))

        # Update PPO policy
        if episode >= 1:  # Start updating after first episode
            ppo_agent.update_ppo(
                gamma=0.99,
                lam=0.97,
                clip_eps=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
                num_epochs=5,
                batch_size=64
            )

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            print(f"\n   Episode {episode+1:3d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Avg (last 10): {np.mean(recent_rewards):7.2f} | "
                  f"Length: {len(states)}")

    # Results
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nInitial reward (episode 1): {episode_rewards[0]:.2f}")
    print(f"Final reward (episode {num_episodes}): {episode_rewards[-1]:.2f}")
    print(f"Average reward (last 20 episodes): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")

    # Check if learning occurred
    initial_avg = np.mean(episode_rewards[:10])
    final_avg = np.mean(episode_rewards[-20:])
    improvement = final_avg - initial_avg

    print(f"\nLearning progress:")
    print(f"  Initial avg (first 10): {initial_avg:.2f}")
    print(f"  Final avg (last 20): {final_avg:.2f}")
    print(f"  Improvement: {improvement:.2f} ({improvement/abs(initial_avg)*100:.1f}%)")

    # Plot results
    print("\n5. Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    # Moving average
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg,
                'r-', linewidth=2, label=f'Moving Average ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Progress on SimplifiedWalker2d')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.6, label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg_len,
                'r-', linewidth=2, label=f'Moving Average ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Length Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_simplified_walker_training.png', dpi=150, bbox_inches='tight')
    print(f"   Plot saved: ppo_simplified_walker_training.png")

    # Test final policy
    print("\n6. Testing final policy...")
    test_states, test_probs, test_actions, test_reward = ppo_agent.generate_session_lax(
        args=args,
        state_train=None,
        D_demo=D_demo,
        iteration=999  # Different seed
    )

    print(f"   Test reward: {test_reward:.2f}")
    print(f"   Test length: {len(test_states)}")

    # Success criteria
    print("\n" + "="*70)
    print("Success Criteria:")
    print("="*70)

    success = True

    # Criterion 1: Learning occurred
    if improvement > 0:
        print("✓ Learning occurred (improvement > 0)")
    else:
        print("✗ No learning detected")
        success = False

    # Criterion 2: Final performance
    if final_avg > initial_avg:
        print(f"✓ Final performance better than initial ({final_avg:.2f} > {initial_avg:.2f})")
    else:
        print(f"✗ Final performance not better than initial")
        success = False

    # Criterion 3: Reasonable rewards
    if final_avg > -100:  # Walker2d expert gets ~60-80, we expect at least positive progress
        print(f"✓ Achieved reasonable rewards (final avg: {final_avg:.2f})")
    else:
        print(f"✗ Rewards still very low (final avg: {final_avg:.2f})")
        success = False

    if success:
        print("\n" + "="*70)
        print("✓✓✓ TEST PASSED: PPO successfully learned on SimplifiedWalker2d! ✓✓✓")
        print("="*70)
        print("\nPPO is ready to be integrated with GCL in main_simplified.py")
    else:
        print("\n" + "="*70)
        print("✗✗✗ TEST FAILED: PPO did not learn effectively ✗✗✗")
        print("="*70)
        print("\nConsider:")
        print("  - Adjusting hyperparameters (learning rate, clip_eps, etc.)")
        print("  - Increasing training episodes")
        print("  - Checking reward function implementation")

    return success, episode_rewards, ppo_agent


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting PPO SimplifiedWalker2d Unit Test")
    print("="*70)
    print("\nThis test will:")
    print("  1. Create a SimplifiedWalker2d environment")
    print("  2. Initialize a PPO agent")
    print("  3. Train the agent for 100 episodes")
    print("  4. Verify learning occurred")
    print("  5. Save training plots")
    print("\nExpected runtime: ~5-10 minutes")
    print()

    success, rewards, agent = test_ppo_simplified_walker()

    if success:
        print("\n✓ PPO is ready for GCL integration!")
    else:
        print("\n✗ PPO needs debugging before GCL integration")

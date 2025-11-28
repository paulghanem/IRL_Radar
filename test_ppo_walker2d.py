"""
Unit test for PPO on Walker2d-v4

This test verifies that PPO can learn a good policy on Walker2d-v4
before integrating it with GCL in the main training loop.
"""

import argparse
import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax.training import train_state
import mujoco
from mujoco import mjx
import gymnasium as gym
from tqdm import tqdm

from src.control.PPO import PPOPolicy, policy_model, critic_model
from src.control.dynamics import get_step_model
from utils.helpers import CustomTerminationWrapper


class Args:
    """Configuration for PPO testing on Walker2d"""
    def __init__(self):
        self.seed = 42
        self.gym_env = "Walker2d"
        self.rollout_length = 512  # Reduced from 2048 to avoid memory issues
        self.ppo_epochs = 5  # Reduced from 10
        self.ppo_batch_size = 64
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.lr = 3e-4


def test_ppo_walker2d():
    """Test PPO learning on Walker2d-v4"""
    print("="*70)
    print("PPO Walker2d-v4 Unit Test")
    print("="*70)

    args = Args()

    # ========== SET UP WALKER2D ENVIRONMENT ==========
    print("\n1. Setting up Walker2d-v4 environment...")

    assets_dir = "assets"
    env_xml = "walker2d.xml"
    frame_skip = 4
    dt = 0.002

    model_path = os.path.join(assets_dir, env_xml)
    model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(model)

    max_frames = args.rollout_length
    env = CustomTerminationWrapper(
        gym.make("Walker2d-v4", exclude_current_positions_from_observation=False, render_mode=None),
        max_steps=max_frames
    )

    obs, info = env.reset(seed=args.seed)
    args.a_dim = env.action_space.shape[0]
    args.s_dim = env.observation_space.shape[0]

    print(f"   State dim: {args.s_dim}")
    print(f"   Action dim: {args.a_dim}")
    print(f"   Frame skip: {frame_skip}")
    print(f"   dt: {dt}")
    print(f"   Effective dt: {dt * frame_skip}")

    # ========== INITIALIZE PPO NETWORKS ==========
    print("\n2. Initializing PPO networks...")

    # Actor network
    model_p = policy_model(action_dim=args.a_dim)
    dummy_input = jnp.zeros((1, args.s_dim))
    init_rng = jax.random.key(args.seed)

    params_p = model_p.init(init_rng, dummy_input)['params']
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=args.lr)
    )
    state_train_p = train_state.TrainState.create(
        apply_fn=model_p.apply,
        params=params_p,
        tx=tx
    )

    # Critic network
    model_c = critic_model()
    params_c = model_c.init(init_rng, dummy_input)['params']
    tx_c = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=args.lr)
    )
    state_train_c = train_state.TrainState.create(
        apply_fn=model_c.apply,
        params=params_c,
        tx=tx_c
    )

    print(f"   Actor network: {args.s_dim} -> 64 -> 64 -> {args.a_dim}")
    print(f"   Critic network: {args.s_dim} -> 64 -> 64 -> 1")

    # ========== CREATE PPO POLICY ==========
    print("\n3. Creating PPO policy...")

    dynamics_fn = get_step_model(args.gym_env, env)
    policy = PPOPolicy(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        mjx_model=mjx_model,
        dynamics=dynamics_fn,
        policy_model=state_train_p,
        policy_net=model_p,
        value_fn=state_train_c,
        args=args,
        rollout_length=args.rollout_length
    )

    print("   PPO policy created successfully")

    # ========== TRAINING LOOP ==========
    print("\n4. Training PPO...")
    print(f"   Episodes: 30")
    print(f"   Steps per rollout: {args.rollout_length}")
    print(f"   PPO epochs per update: {args.ppo_epochs}")
    print()

    num_episodes = 30
    episode_rewards = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        obs, info = env.reset(seed=args.seed + episode)
        x0 = obs.reshape((1, -1))

        # Generate rollout
        policy.generate_session_lax(args, x0, frame_skip, dt)

        # Get buffer data
        states, actions, rewards, dones, log_probs_old, next_states = policy.buffer.get()

        # Convert to arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards).flatten()
        dones = dones.flatten()
        log_probs_old = jnp.array(log_probs_old).flatten()

        total_reward = float(jnp.sum(rewards))
        episode_rewards.append(total_reward)

        # Update PPO policy
        policy.update_ppo(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs_old=log_probs_old,
            next_states=next_states,
            num_epochs=args.ppo_epochs,
            gamma=args.gamma,
            clip_eps=args.clip_eps
        )

        # Print progress every 5 episodes
        if (episode + 1) % 5 == 0:
            recent_rewards = episode_rewards[-5:]
            print(f"\n   Episode {episode+1:3d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Avg (last 5): {np.mean(recent_rewards):7.2f}")

    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nInitial reward (episode 1): {episode_rewards[0]:.2f}")
    print(f"Final reward (episode {num_episodes}): {episode_rewards[-1]:.2f}")
    print(f"Average reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")

    # Check if learning occurred
    initial_avg = np.mean(episode_rewards[:5])
    final_avg = np.mean(episode_rewards[-10:])
    improvement = final_avg - initial_avg

    print(f"\nLearning progress:")
    print(f"  Initial avg (first 5): {initial_avg:.2f}")
    print(f"  Final avg (last 10): {final_avg:.2f}")
    print(f"  Improvement: {improvement:.2f}")
    if initial_avg != 0:
        print(f"  Relative improvement: {improvement/abs(initial_avg)*100:.1f}%")

    # ========== PLOT RESULTS ==========
    print("\n5. Generating plots...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot rewards
    ax.plot(episode_rewards, alpha=0.6, label='Episode Reward', marker='o', markersize=3)

    # Moving average
    window = 5
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), moving_avg,
                'r-', linewidth=2, label=f'Moving Average ({window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('PPO Training Progress on Walker2d-v4')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_walker2d_training.png', dpi=150, bbox_inches='tight')
    print(f"   Plot saved: ppo_walker2d_training.png")

    # ========== EVALUATION ==========
    print("\n6. Evaluating final policy...")

    eval_rewards = []
    for eval_ep in range(5):
        obs, info = env.reset(seed=args.seed + 1000 + eval_ep)
        x0 = obs.reshape((1, -1))

        policy.generate_session_lax(args, x0, frame_skip, dt)
        states, actions, rewards, dones, log_probs_old, next_states = policy.buffer.get()

        eval_reward = float(jnp.sum(rewards))
        eval_rewards.append(eval_reward)

    print(f"   Evaluation rewards (5 episodes): {[f'{r:.2f}' for r in eval_rewards]}")
    print(f"   Average evaluation reward: {np.mean(eval_rewards):.2f}")

    # ========== SUCCESS CRITERIA ==========
    print("\n" + "="*70)
    print("Success Criteria:")
    print("="*70)

    success = True

    # Criterion 1: Learning occurred
    if improvement > 0:
        print(f"✓ Learning occurred (improvement: {improvement:.2f} > 0)")
    else:
        print(f"✗ No learning detected (improvement: {improvement:.2f})")
        success = False

    # Criterion 2: Final performance better than initial
    if final_avg > initial_avg:
        print(f"✓ Final performance better than initial ({final_avg:.2f} > {initial_avg:.2f})")
    else:
        print(f"✗ Final performance not better than initial")
        success = False

    # Criterion 3: Reasonable final rewards
    # Walker2d expert typically gets 60-80, we expect at least some positive progress
    if final_avg > -50:
        print(f"✓ Achieved reasonable rewards (final avg: {final_avg:.2f} > -50)")
    else:
        print(f"✗ Rewards still very low (final avg: {final_avg:.2f})")
        success = False

    if success:
        print("\n" + "="*70)
        print("✓✓✓ TEST PASSED: PPO successfully learned on Walker2d-v4! ✓✓✓")
        print("="*70)
        print("\nPPO is ready to be integrated with GCL for inverse RL!")
    else:
        print("\n" + "="*70)
        print("✗✗✗ TEST FAILED: PPO did not learn effectively ✗✗✗")
        print("="*70)
        print("\nConsider:")
        print("  - Running more training episodes")
        print("  - Adjusting hyperparameters (learning rate, clip_eps, etc.)")
        print("  - Checking reward function implementation")

    return success, episode_rewards, policy


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting PPO Walker2d-v4 Unit Test")
    print("="*70)
    print("\nThis test will:")
    print("  1. Create a Walker2d-v4 environment (MuJoCo + MJX)")
    print("  2. Initialize a PPO agent")
    print("  3. Train the agent for 50 episodes")
    print("  4. Verify learning occurred")
    print("  5. Save training plots")
    print("\nExpected runtime: ~10-15 minutes")
    print()

    success, rewards, agent = test_ppo_walker2d()

    if success:
        print("\n✓ PPO is ready for GCL integration!")
    else:
        print("\n✗ PPO needs debugging before GCL integration")

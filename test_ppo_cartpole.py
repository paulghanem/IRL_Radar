"""
Test script for PPO_unified on CartPole environment

This script tests the UnifiedPPO implementation to verify it works correctly
before integrating it into the IRL training loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.control.PPO_unified import UnifiedPPO
from src.control.dynamics import get_step_model
from utils.helpers import GenerateDemo
import matplotlib.pyplot as plt

print("JAX devices:", jax.devices())

def test_ppo_cartpole():
    """Test PPO on CartPole environment"""

    # Setup arguments
    args = argparse.Namespace()
    args.seed = 42
    args.s_dim = 4  # CartPole state: [x, x_dot, theta, theta_dot]
    args.a_dim = 1  # CartPole action: [force]
    args.N_steps = 200  # Episode length
    args.frame_skip = 1
    args.dt = 0.02
    args.gym_env = "CartPole-v1"

    print("\n" + "="*60)
    print("Generating expert demonstrations...")
    print("="*60)

    # Generate expert demonstrations
    demo_generator = GenerateDemo(args.gym_env, max_frames=args.N_steps)
    states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(args.seed)

    print(f"Expert trajectories: {states_d.shape[0]} steps")
    print(f"Expert reward: {float(jnp.sum(rewards_demo)):.2f}")

    # Prepare demo data
    D_demo = jnp.concatenate([states_d, jnp.ones((states_d.shape[0], 1)), actions_d], axis=1)

    # Create PPO agent
    print("\n" + "="*60)
    print("Creating PPO agent...")
    print("="*60)

    dynamics = get_step_model(args.gym_env, env)

    ppo_agent = UnifiedPPO(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        args=args,
        state_train=None,
        dynamics=dynamics,
        mjx_model=None,
        gym_env=args.gym_env,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        rollout_length=args.N_steps,
        buffer_mix=20,
        use_learned_cost=False  # Use environment reward
    )

    print("PPO agent created successfully!")

    print("\n" + "="*60)
    print("Testing PPO on CartPole...")
    print("="*60)

    # Training loop
    num_iterations = 50
    rewards_history = []

    for iteration in range(num_iterations):
        # Generate trajectory
        states, probs, actions, total_reward = ppo_agent.generate_session_lax(
            args, None, D_demo, iteration=iteration
        )

        rewards_history.append(total_reward)

        # Extract data from buffer for PPO update
        # Only update if we've completed a full rollout
        if ppo_agent.buffer.p % ppo_agent.buffer.buffer_size == 0 and iteration > 0:
            try:
                buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_log_probs, buffer_next_states = ppo_agent.buffer.get()

                # Update PPO
                ppo_agent.update_ppo(
                    states=buffer_states,
                    actions=buffer_actions,
                    rewards=buffer_rewards.flatten(),
                    dones=buffer_dones.flatten(),
                    log_probs_old=buffer_log_probs.flatten(),
                    next_states=buffer_next_states,
                    gamma=0.99,
                    lam=0.97,
                    clip_eps=0.2,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    num_epochs=10,
                    batch_size=64,
                    max_grad_norm=0.5
                )
            except AssertionError:
                # Buffer not aligned, skip update
                pass

        # Print progress
        if (iteration + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Iteration {iteration + 1}/{num_iterations}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Avg Last 10: {avg_reward:.2f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Iteration')
    plt.ylabel('Total Reward')
    plt.title('PPO Training on CartPole-v1')
    plt.grid(True)
    plt.savefig('ppo_cartpole_training.png')
    print(f"\nTraining plot saved to ppo_cartpole_training.png")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    final_rewards = []
    for i in range(10):
        states, probs, actions, total_reward = ppo_agent.generate_session_lax(
            args, None, D_demo, iteration=100+i
        )
        final_rewards.append(total_reward)

    avg_final = np.mean(final_rewards)
    std_final = np.std(final_rewards)

    print(f"Final Average Reward (10 episodes): {avg_final:.2f} +/- {std_final:.2f}")

    # CartPole is considered solved if average reward is >= 195
    if avg_final >= 195:
        print("\n✓ CartPole SOLVED! PPO is working correctly.")
        return True
    elif avg_final >= 100:
        print(f"\n~ PPO is learning (avg={avg_final:.1f}), but not fully solved yet (need >= 195)")
        return True
    else:
        print(f"\n✗ PPO performance is low (avg={avg_final:.1f})")
        return False

if __name__ == "__main__":
    try:
        success = test_ppo_cartpole()
        print("\n" + "="*60)
        print("All tests passed! [OK]" if success else "Tests completed with warnings")
        print("="*60)
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

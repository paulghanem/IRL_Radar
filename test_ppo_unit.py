"""
Unit Test for Unified PPO Implementation

This script tests the PPO implementation by training on CartPole-v1 from scratch.
Success criteria: Average reward > 150 within 100 episodes.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from dataclasses import dataclass

from src.control.PPO_unified import UnifiedPPO
from src.control.dynamics import get_action_space, get_action_cov, get_step_model


@dataclass
class TestArgs:
    """Test configuration"""
    seed: int = 42
    gym_env: str = "CartPole-v1"
    s_dim: int = 4
    a_dim: int = 1
    N_steps: int = 200  # Episode length
    frame_skip: int = 1
    dt: float = 0.02


def create_dummy_demo(state_dim, action_dim, num_steps=200):
    """Create dummy expert demonstrations for testing"""
    states = jnp.zeros((num_steps, state_dim))
    actions = jnp.zeros((num_steps, action_dim))
    D_demo = jnp.concatenate([states, jnp.ones((num_steps, 1)), actions], axis=1)
    return D_demo


def test_ppo_cartpole():
    """Test PPO on CartPole environment"""
    print("=" * 80)
    print("PPO Unit Test: Training CartPole-v1 from Scratch")
    print("=" * 80)

    # Configuration
    args = TestArgs()

    # Simple CartPole dynamics
    def cartpole_dynamics(state, action):
        """
        Simple CartPole dynamics for testing

        State: [x, x_dot, theta, theta_dot]
        Action: force (scalar)
        """
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masscart + masspole
        length = 0.5  # half-pole length
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02  # time step

        # Unpack state
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]

        # Apply force (action is already in [-1, 1] from tanh)
        force = action[0] * force_mag

        # Physics equations
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        temp = (force + polemass_length * theta_dot ** 2 * sin_theta) / total_mass
        thetaacc = (gravity * sin_theta - cos_theta * temp) / (
            length * (4.0 / 3.0 - masspole * cos_theta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * cos_theta / total_mass

        # Euler integration
        x_new = x + tau * x_dot
        x_dot_new = x_dot + tau * xacc
        theta_new = theta + tau * theta_dot
        theta_dot_new = theta_dot + tau * thetaacc

        return jnp.array([x_new, x_dot_new, theta_new, theta_dot_new])

    # Create PPO agent
    print("\nInitializing PPO agent...")
    ppo = UnifiedPPO(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        args=args,
        dynamics=cartpole_dynamics,
        gym_env=args.gym_env,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        rollout_length=args.N_steps,
        use_learned_cost=False  # Use environment reward
    )
    print("PPO agent initialized successfully!")

    # Create dummy demo data
    D_demo = create_dummy_demo(args.s_dim, args.a_dim, args.N_steps)

    # Training loop
    num_episodes = 100
    rewards_history = []

    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 80)

    for episode in range(num_episodes):
        # Generate trajectory
        states, probs, actions, total_reward = ppo.generate_session_lax(
            args, None, D_demo, iteration=episode
        )

        rewards_history.append(total_reward)

        # Get data from buffer for PPO update
        buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_logps, buffer_next_states = ppo.buffer.get()

        # Update PPO policy
        ppo.update_ppo(
            states=buffer_states,
            actions=buffer_actions,
            rewards=buffer_rewards,
            dones=buffer_dones,
            log_probs_old=buffer_logps,
            next_states=buffer_next_states,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            num_epochs=4,
            batch_size=64,
            max_grad_norm=0.5
        )

        # Print progress
        if (episode + 1) % 10 == 0:
            recent_rewards = rewards_history[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1:3d} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Avg (last 10): {avg_reward:6.2f}")

    # Final evaluation
    print("-" * 80)
    print("\nFinal Results:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Final 10-episode average: {np.mean(rewards_history[-10:]):.2f}")
    print(f"  Best episode reward: {np.max(rewards_history):.2f}")
    print(f"  Worst episode reward: {np.min(rewards_history):.2f}")

    # Success criterion
    final_avg = np.mean(rewards_history[-10:])
    if final_avg > 150:
        print(f"\n[PASS] SUCCESS: Average reward ({final_avg:.2f}) > 150")
        return True
    else:
        print(f"\n[FAIL] FAILURE: Average reward ({final_avg:.2f}) <= 150")
        print("  Note: PPO may need more training or hyperparameter tuning")
        return False


def test_ppo_interface():
    """Test that PPO has the same interface as MPPI"""
    print("\n" + "=" * 80)
    print("Testing PPO Interface Compatibility with MPPI")
    print("=" * 80)

    args = TestArgs()

    def dummy_dynamics(state, action):
        return state

    ppo = UnifiedPPO(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        args=args,
        dynamics=dummy_dynamics,
        gym_env=args.gym_env
    )

    # Test that generate_session_lax exists and has correct signature
    D_demo = create_dummy_demo(args.s_dim, args.a_dim, 10)

    try:
        result = ppo.generate_session_lax(args, None, D_demo, iteration=0)
        states, probs, actions, total_reward = result

        print("\n[PASS] generate_session_lax() method exists")
        print(f"[PASS] Returns 4 values: states, probs, actions, reward")
        print(f"  - States shape: {np.array(states).shape}")
        print(f"  - Probs shape: {np.array(probs).shape}")
        print(f"  - Actions shape: {np.array(actions).shape}")
        print(f"  - Total reward: {total_reward}")

        return True
    except Exception as e:
        print(f"\n[FAIL] FAILURE: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("UNIFIED PPO UNIT TESTS")
    print("=" * 80)

    # Test 1: Interface compatibility
    test1_passed = test_ppo_interface()

    # Test 2: Training from scratch
    test2_passed = test_ppo_cartpole()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Interface Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"CartPole Training Test: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\nALL TESTS PASSED! PPO is ready to replace MPPI.")
    else:
        print("\nSOME TESTS FAILED. Please review the implementation.")

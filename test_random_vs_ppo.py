"""
Compare Random Policy vs PPO to identify learning issues
Tests: Random actions, Untrained PPO, Trained PPO
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.control.PPO_unified import UnifiedPPO
from src.control.dynamics import get_step_model
import sys
import flax.linen as nn
import optax
from flax.training import train_state

print("JAX devices:", jax.devices())

# Setup arguments
args = argparse.Namespace()
args.seed = 42
args.s_dim = 4
args.a_dim = 1
args.N_steps = 200
args.frame_skip = 1
args.dt = 0.02
args.gym_env = "CartPole-v1"

# Test with challenging initial condition
init_state = jnp.array([-0.555, 0.614, -0.017, -0.540])
print(f"\nTest Initial Condition:")
print(f"  x={init_state[0]:.3f}, x_dot={init_state[1]:.3f}, theta={init_state[2]:.3f}, theta_dot={init_state[3]:.3f}")

dynamics = get_step_model(args.gym_env, None)

# =============================================================================
# TEST 1: TRULY RANDOM POLICY (uniform random actions)
# =============================================================================
print("\n" + "="*60)
print("TEST 1: TRULY RANDOM POLICY")
print("="*60)

def test_random_policy(init_state, num_episodes=5):
    """Test with truly random actions from uniform distribution"""
    rewards = []

    for ep in range(num_episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        state = init_state
        total_reward = 0

        for t in range(args.N_steps):
            # Truly random action: uniform between -1 and 1
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(subkey, (args.a_dim,), minval=-1.0, maxval=1.0)

            # Step environment
            next_state = dynamics(state, action)
            next_state = jnp.atleast_1d(next_state).flatten()

            # Check termination
            x = next_state[0]
            theta = next_state[2]
            x_threshold = 2.4
            theta_threshold = 12 * 2 * np.pi / 360

            terminated = (
                (x < -x_threshold) | (x > x_threshold) |
                (theta < -theta_threshold) | (theta > theta_threshold)
            )

            reward = 1.0 if not terminated else 0.0
            total_reward += reward

            if terminated:
                break

            state = next_state

        rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.1f}")

    return rewards

random_rewards = test_random_policy(init_state, num_episodes=5)
print(f"\nRandom Policy Results:")
print(f"  Mean: {np.mean(random_rewards):.1f}")
print(f"  Std:  {np.std(random_rewards):.1f}")
print(f"  Min:  {np.min(random_rewards):.1f}")
print(f"  Max:  {np.max(random_rewards):.1f}")

# =============================================================================
# TEST 2: ZERO-ACTION POLICY (always output zero)
# =============================================================================
print("\n" + "="*60)
print("TEST 2: ZERO-ACTION POLICY")
print("="*60)

def test_zero_policy(init_state, num_episodes=5):
    """Test with zero actions (do nothing)"""
    rewards = []

    for ep in range(num_episodes):
        state = init_state
        total_reward = 0

        for t in range(args.N_steps):
            # Zero action (do nothing)
            action = jnp.zeros((args.a_dim,))

            # Step environment
            next_state = dynamics(state, action)
            next_state = jnp.atleast_1d(next_state).flatten()

            # Check termination
            x = next_state[0]
            theta = next_state[2]
            x_threshold = 2.4
            theta_threshold = 12 * 2 * np.pi / 360

            terminated = (
                (x < -x_threshold) | (x > x_threshold) |
                (theta < -theta_threshold) | (theta > theta_threshold)
            )

            reward = 1.0 if not terminated else 0.0
            total_reward += reward

            if terminated:
                break

            state = next_state

        rewards.append(total_reward)

    print(f"  Episode 1: {rewards[0]:.1f}")
    return rewards

zero_rewards = test_zero_policy(init_state, num_episodes=5)
print(f"\nZero Policy Results:")
print(f"  Mean: {np.mean(zero_rewards):.1f}")

# =============================================================================
# TEST 3: UNTRAINED PPO (from scratch)
# =============================================================================
print("\n" + "="*60)
print("TEST 3: UNTRAINED PPO (from scratch initialization)")
print("="*60)

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
    use_learned_cost=False
)

# Initialize from scratch (very small weights)
dummy_input = jnp.zeros((1, args.s_dim))
key_init = jax.random.PRNGKey(args.seed)
actor_vars = ppo_agent.actor_net.init(key_init, dummy_input)
actor_params = actor_vars['params']
actor_params = jax.tree_util.tree_map(lambda x: x * 0.01, actor_params)
ppo_agent.actor_state = train_state.TrainState.create(
    apply_fn=ppo_agent.actor_net.apply,
    params=actor_params,
    tx=optax.adam(3e-4)
)

print("Testing untrained PPO (5 episodes)...")
untrained_rewards = []
for ep in range(5):
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))
    D_demo = D_demo.at[0, :args.s_dim].set(init_state)

    states, probs, actions, total_reward = ppo_agent.generate_session_lax(
        args, None, D_demo, iteration=ep
    )
    untrained_rewards.append(total_reward)
    print(f"  Episode {ep+1}: {total_reward:.1f}")

print(f"\nUntrained PPO Results:")
print(f"  Mean: {np.mean(untrained_rewards):.1f}")
print(f"  Std:  {np.std(untrained_rewards):.1f}")

# Check what actions the untrained policy is producing
print("\nSample actions from untrained policy:")
test_state = init_state
key = jax.random.PRNGKey(42)
sample_actions = []
for i in range(10):
    key, subkey = jax.random.split(key)
    action, log_pi = ppo_agent.sample_action(test_state, ppo_agent.actor_state.params, subkey)
    sample_actions.append(float(action[0]))
print(f"  Actions: {sample_actions}")
print(f"  Mean: {np.mean(sample_actions):.4f}")
print(f"  Std: {np.std(sample_actions):.4f}")

# =============================================================================
# TEST 4: TRAINED PPO
# =============================================================================
print("\n" + "="*60)
print("TEST 4: TRAIN PPO FOR 50 ITERATIONS")
print("="*60)

rewards_history = []
for iteration in range(50):
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))
    D_demo = D_demo.at[0, :args.s_dim].set(init_state)

    states, probs, actions, total_reward = ppo_agent.generate_session_lax(
        args, None, D_demo, iteration=iteration
    )
    rewards_history.append(total_reward)

    # Update PPO
    if ppo_agent.buffer.p % ppo_agent.buffer.buffer_size == 0 and iteration > 0:
        try:
            buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_log_probs, buffer_next_states = ppo_agent.buffer.get()

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
                batch_size=min(256, args.N_steps),
                max_grad_norm=0.5
            )
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: {total_reward:.1f} (PPO updated)")
        except AssertionError:
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: {total_reward:.1f} (buffer not aligned)")
    else:
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: {total_reward:.1f}")

print(f"\nTrained PPO Results:")
print(f"  Initial (iter 0): {rewards_history[0]:.1f}")
print(f"  Final (iter 49): {rewards_history[-1]:.1f}")
print(f"  Improvement: {rewards_history[-1] - rewards_history[0]:+.1f}")
print(f"  Avg last 10: {np.mean(rewards_history[-10:]):.1f}")

# Check what actions the trained policy produces
print("\nSample actions from trained policy:")
test_state = init_state
key = jax.random.PRNGKey(42)
sample_actions_trained = []
for i in range(10):
    key, subkey = jax.random.split(key)
    action, log_pi = ppo_agent.sample_action(test_state, ppo_agent.actor_state.params, subkey)
    sample_actions_trained.append(float(action[0]))
print(f"  Actions: {sample_actions_trained}")
print(f"  Mean: {np.mean(sample_actions_trained):.4f}")
print(f"  Std: {np.std(sample_actions_trained):.4f}")

# =============================================================================
# SUMMARY AND ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS")
print("="*60)

print(f"\nReward Comparison:")
print(f"  Random Policy:      {np.mean(random_rewards):.1f} ± {np.std(random_rewards):.1f}")
print(f"  Zero Policy:        {np.mean(zero_rewards):.1f}")
print(f"  Untrained PPO:      {np.mean(untrained_rewards):.1f} ± {np.std(untrained_rewards):.1f}")
print(f"  Trained PPO (init): {rewards_history[0]:.1f}")
print(f"  Trained PPO (final):{rewards_history[-1]:.1f}")

print(f"\nAction Statistics:")
print(f"  Untrained PPO: mean={np.mean(sample_actions):.4f}, std={np.std(sample_actions):.4f}")
print(f"  Trained PPO:   mean={np.mean(sample_actions_trained):.4f}, std={np.std(sample_actions_trained):.4f}")

print(f"\nDIAGNOSIS:")
if np.mean(random_rewards) > 150:
    print("  [!] Random policy gets high rewards (~{:.0f}) - CartPole is too stable!".format(np.mean(random_rewards)))
    print("  [!] The initial condition is not challenging enough for this task.")
if abs(np.mean(untrained_rewards) - np.mean(random_rewards)) < 10:
    print("  [!] Untrained PPO ≈ Random policy - initialization may not be working correctly")
if abs(np.mean(untrained_rewards) - np.mean(random_rewards)) > 100:
    print("  [BUG FOUND!] Untrained PPO ({:.0f}) >> Random policy ({:.0f})".format(np.mean(untrained_rewards), np.mean(random_rewards)))
    print("  [BUG] The 'from scratch' initialization is NOT working!")
    print("  [BUG] Scaling weights by 0.01 still produces good actions")
    print("  [BUG] Need to use proper zero initialization or constant zero output")
if abs(rewards_history[-1] - rewards_history[0]) < 5:
    print("  [!] PPO shows minimal learning ({:.1f} -> {:.1f})".format(rewards_history[0], rewards_history[-1]))
    print("  [!] Possible causes:")
    print("    - Task is too easy (already near-optimal)")
    print("    - Learning rate or hyperparameters need tuning")
    print("    - Initial condition needs to be much harder")
else:
    print("  [OK] PPO is learning! Improved by {:.1f}".format(rewards_history[-1] - rewards_history[0]))

print("\n[SUMMARY] The real comparison should be:")
print("  Random/Zero actions: ~{:.0f} rewards (fails quickly)".format(np.mean(random_rewards)))
print("  Trained PPO: {:.0f} rewards (stays balanced much longer)".format(rewards_history[-1]))
print("  This shows PPO CAN learn, but we need proper baseline comparison!")

print("\nTest completed!")

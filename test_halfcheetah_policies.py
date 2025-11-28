"""
Test Random, Zero, and PPO policies on HalfCheetah-v4
Using the SAME generate_session_lax structure for fair comparison
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.control.PPO_unified import UnifiedPPO
import sys
import optax
from flax.training import train_state
import mujoco
from mujoco import mjx

print("JAX devices:", jax.devices())

# Setup arguments for HalfCheetah-v4
args = argparse.Namespace()
args.seed = 42
args.s_dim = 18  # HalfCheetah state dimension (qpos: 9, qvel: 9)
args.a_dim = 6   # HalfCheetah action dimension
args.N_steps = 200  # Episode length
args.frame_skip = 5
args.dt = 0.05
args.gym_env = "HalfCheetah-v4"

print(f"\nEnvironment: {args.gym_env}")
print(f"State dim: {args.s_dim}, Action dim: {args.a_dim}")
print(f"Episode length: {args.N_steps} steps")

# Load MuJoCo model
try:
    mj_model = mujoco.MjModel.from_xml_path("assets/half_cheetah.xml")
    mjx_model = mjx.put_model(mj_model)
    print("MuJoCo model loaded successfully")
except Exception as e:
    print(f"Error loading MuJoCo model: {e}")
    sys.exit(1)

# =============================================================================
# Create PPO agent
# =============================================================================
ppo_agent = UnifiedPPO(
    state_dim=args.s_dim,
    action_dim=args.a_dim,
    args=args,
    state_train=None,
    dynamics=None,  # MuJoCo handles dynamics
    mjx_model=mjx_model,
    gym_env=args.gym_env,
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=1e-3,
    rollout_length=args.N_steps,
    buffer_mix=20,
    use_learned_cost=False
)

# Save the original sample_action method
original_sample_action = ppo_agent.sample_action

# =============================================================================
# TEST 1: RANDOM POLICY (using generate_session_lax)
# =============================================================================
print("\n" + "="*60)
print("TEST 1: RANDOM POLICY (through generate_session_lax)")
print("="*60)

def random_sample_action(state, params, key):
    """Sample random actions uniformly from [-1, 1]"""
    action = jax.random.uniform(key, (args.a_dim,), minval=-1.0, maxval=1.0)
    log_pi = jnp.zeros((1,))  # Dummy log prob
    return action, log_pi

# Temporarily replace sample_action with random policy
ppo_agent.sample_action = random_sample_action

random_rewards = []
for ep in range(5):
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))
    # D_demo initial state will be set by reset in generate_session_lax

    states, probs, actions, total_reward = ppo_agent.generate_session_lax(
        args, None, D_demo, iteration=ep
    )
    random_rewards.append(total_reward)
    print(f"  Episode {ep+1}: {total_reward:.1f}")

print(f"\nRandom Policy Results:")
print(f"  Mean: {np.mean(random_rewards):.1f}")
print(f"  Std:  {np.std(random_rewards):.1f}")

# =============================================================================
# TEST 2: ZERO POLICY (using generate_session_lax)
# =============================================================================
print("\n" + "="*60)
print("TEST 2: ZERO POLICY (through generate_session_lax)")
print("="*60)

def zero_sample_action(state, params, key):
    """Always return zero actions"""
    action = jnp.zeros((args.a_dim,))
    log_pi = jnp.zeros((1,))  # Dummy log prob
    return action, log_pi

# Replace with zero policy
ppo_agent.sample_action = zero_sample_action

zero_rewards = []
for ep in range(3):
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))

    states, probs, actions, total_reward = ppo_agent.generate_session_lax(
        args, None, D_demo, iteration=ep
    )
    zero_rewards.append(total_reward)

print(f"  Episode 1: {zero_rewards[0]:.1f}")

print(f"\nZero Policy Results:")
print(f"  Mean: {np.mean(zero_rewards):.1f}")

# =============================================================================
# TEST 3: UNTRAINED PPO (using generate_session_lax)
# =============================================================================
print("\n" + "="*60)
print("TEST 3: UNTRAINED PPO (through generate_session_lax)")
print("="*60)

# Restore original sample_action
ppo_agent.sample_action = original_sample_action

# Initialize PPO from scratch (zero out the final layer)
dummy_input = jnp.zeros((1, args.s_dim))
key_init = jax.random.PRNGKey(args.seed)
actor_vars = ppo_agent.actor_net.init(key_init, dummy_input)
actor_params = actor_vars['params']

# Zero out only the final Dense layer
if 'Dense_2' in actor_params:
    actor_params['Dense_2']['kernel'] = jnp.zeros_like(actor_params['Dense_2']['kernel'])
    actor_params['Dense_2']['bias'] = jnp.zeros_like(actor_params['Dense_2']['bias'])

ppo_agent.actor_state = train_state.TrainState.create(
    apply_fn=ppo_agent.actor_net.apply,
    params=actor_params,
    tx=optax.adam(3e-4)
)

print("Testing untrained PPO (3 episodes)...")
untrained_rewards = []
for ep in range(3):
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))

    states, probs, actions, total_reward = ppo_agent.generate_session_lax(
        args, None, D_demo, iteration=ep
    )
    untrained_rewards.append(total_reward)
    print(f"  Episode {ep+1}: {total_reward:.1f}")

print(f"\nUntrained PPO Results:")
print(f"  Mean: {np.mean(untrained_rewards):.1f}")
print(f"  Std:  {np.std(untrained_rewards):.1f}")

# =============================================================================
# TEST 4: TRAIN PPO (using generate_session_lax)
# =============================================================================
print("\n" + "="*60)
print("TEST 4: TRAIN PPO (through generate_session_lax)")
print("="*60)

rewards_history = []
for iteration in range(100):  # More iterations for harder task
    D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))

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
                print(f"  Iteration {iteration}: {total_reward:.1f}")
    else:
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: {total_reward:.1f}")

print(f"\nTrained PPO Results:")
print(f"  Initial (iter 0): {rewards_history[0]:.1f}")
print(f"  Final (iter 99): {rewards_history[-1]:.1f}")
print(f"  Improvement: {rewards_history[-1] - rewards_history[0]:+.1f}")
print(f"  Avg last 10: {np.mean(rewards_history[-10:]):.1f}")
print(f"  Max reward: {np.max(rewards_history):.1f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY - HalfCheetah-v4 (all using generate_session_lax)")
print("="*60)

print(f"\nReward Comparison:")
print(f"  Random Policy:      {np.mean(random_rewards):.1f} +/- {np.std(random_rewards):.1f}")
print(f"  Zero Policy:        {np.mean(zero_rewards):.1f}")
print(f"  Untrained PPO:      {np.mean(untrained_rewards):.1f} +/- {np.std(untrained_rewards):.1f}")
print(f"  Trained PPO:        {rewards_history[-1]:.1f}")

print(f"\nLearning Improvement:")
improvement_from_random = rewards_history[-1] - np.mean(random_rewards)
improvement_from_zero = rewards_history[-1] - np.mean(zero_rewards)
print(f"  Random -> Trained:   {improvement_from_random:+.1f} ({100*improvement_from_random/(abs(np.mean(random_rewards))+0.1):+.0f}%)")
print(f"  Zero -> Trained:     {improvement_from_zero:+.1f} ({100*improvement_from_zero/(abs(np.mean(zero_rewards))+0.1):+.0f}%)")
print(f"  Untrained -> Trained:{rewards_history[-1] - np.mean(untrained_rewards):+.1f}")

print(f"\n[ANALYSIS]")
if improvement_from_random > 100:
    print(f"  SUCCESS! PPO learned significantly better than random")
    print(f"  Improvement: {improvement_from_random:+.0f} rewards")
elif improvement_from_random > 20:
    print(f"  PPO is learning! Improvement: {improvement_from_random:+.1f}")
    print(f"  May benefit from more training iterations")
else:
    print(f"  Minimal learning detected ({improvement_from_random:+.1f} improvement)")
    print(f"  HalfCheetah is a hard task - may need more iterations or tuning")

print("\nTest completed!")

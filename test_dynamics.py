"""
Test to verify dynamics are working correctly
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.control.PPO_unified import UnifiedPPO
from src.control.dynamics import get_step_model

print("JAX devices:", jax.devices())

# Setup arguments
args = argparse.Namespace()
args.seed = 42
args.s_dim = 4
args.a_dim = 1
args.N_steps = 100  # Shorter for debugging
args.frame_skip = 1
args.dt = 0.02
args.gym_env = "CartPole-v1"

print("\nCreating PPO agent...")
dynamics = get_step_model(args.gym_env, None)

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

# Create demo with RANDOM initial condition
key = jax.random.PRNGKey(args.seed)
# Random initial state for CartPole: x, x_dot, theta, theta_dot
# x: -0.5 to 0.5, x_dot: -0.5 to 0.5, theta: -0.2 to 0.2, theta_dot: -0.5 to 0.5
random_init = jax.random.uniform(key, (args.s_dim,), minval=-0.5, maxval=0.5)
random_init = random_init.at[2].set(random_init[2] * 0.4)  # theta: -0.2 to 0.2

D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))
D_demo = D_demo.at[0, :args.s_dim].set(random_init)

print(f"\nRandom initial state: {random_init}")
print("Running rollout and checking state evolution...")

# Generate rollout
states, probs, actions, total_reward = ppo_agent.generate_session_lax(
    args, None, D_demo, iteration=0
)

states_arr = np.array(states)
actions_arr = np.array(actions)

print(f"\nRollout completed!")
print(f"Total reward: {total_reward}")
print(f"\nState statistics:")
print(f"  States shape: {states_arr.shape}")
print(f"  Initial state: {states_arr[0]}")
print(f"  Final state: {states_arr[-1]}")
print(f"  State mean: {states_arr.mean(axis=0)}")
print(f"  State std: {states_arr.std(axis=0)}")
print(f"  State min: {states_arr.min(axis=0)}")
print(f"  State max: {states_arr.max(axis=0)}")

print(f"\nAction statistics:")
print(f"  Actions shape: {actions_arr.shape}")
print(f"  Action mean: {actions_arr.mean()}")
print(f"  Action std: {actions_arr.std()}")
print(f"  Action min: {actions_arr.min()}")
print(f"  Action max: {actions_arr.max()}")

print(f"\nFirst 10 states:")
for i in range(min(10, len(states_arr))):
    print(f"  Step {i}: x={states_arr[i,0]:.4f}, x_dot={states_arr[i,1]:.4f}, "
          f"theta={states_arr[i,2]:.4f}, theta_dot={states_arr[i,3]:.4f}, "
          f"action={actions_arr[i,0]:.4f}")

# Check if states are changing
state_change = np.abs(states_arr[1:] - states_arr[:-1]).max()
print(f"\nMax state change between steps: {state_change:.6f}")

if state_change < 1e-6:
    print("⚠️  WARNING: States are NOT changing! Dynamics might be broken.")
else:
    print("✓ States are changing correctly.")

# Test dynamics directly
print(f"\nTesting dynamics function directly...")
test_state = jnp.array([0.0, 0.0, 0.05, 0.0])
test_action = jnp.array([0.5])
next_state = dynamics(test_state, test_action)
print(f"  Input state: {test_state}")
print(f"  Input action: {test_action}")
print(f"  Output state: {next_state}")
print(f"  State change: {jnp.abs(next_state - test_state).max():.6f}")

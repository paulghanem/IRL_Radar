"""
Basic test of Simplified HalfCheetah dynamics (no MPPI)
"""

import jax
import jax.numpy as jnp
import numpy as np

print("Starting basic HalfCheetah test...")
print("Importing SimplifiedHalfCheetah...")

from src.control.simplified_halfcheetah import SimplifiedHalfCheetah, simplified_halfcheetah_step

print("JAX devices:", jax.devices())
print("\n" + "="*60)
print("Testing Simplified HalfCheetah Dynamics (Basic)")
print("="*60)

# Initialize
print("\nInitializing HalfCheetah...")
cheetah = SimplifiedHalfCheetah()
print(f"  State dim: {cheetah.state_dim}")
print(f"  Action dim: {cheetah.action_dim}")
print(f"  Time step: {cheetah.dt}")

# Test reset
print("\nTesting reset...")
key = jax.random.PRNGKey(42)
state = cheetah.reset(key)
print(f"  Initial state shape: {state.shape}")
print(f"  Initial state: {state}")

# Test single step
print("\nTesting single step...")
action = jnp.array([0.1, 0.2, 0.1, 0.2, 0.05, 0.05])
next_state = cheetah.step(state, action)
print(f"  Next state shape: {next_state.shape}")
print(f"  Next state: {next_state}")

# Test reward computation
print("\nTesting reward computation...")
reward = cheetah.compute_reward(state, action, next_state)
print(f"  Reward: {reward}")

# Test trajectory
print("\nTesting 10-step trajectory...")
state = cheetah.reset(key)
total_reward = 0.0

for i in range(10):
    # Simple forward policy
    action = jnp.array([0.3, 0.5, 0.2, 0.3, 0.5, 0.2])
    next_state = cheetah.step(state, action)
    reward = cheetah.compute_reward(state, action, next_state)
    total_reward += reward

    if i < 3:  # Print first 3 steps
        print(f"  Step {i+1}: reward={reward:.3f}, x_vel={next_state[8]:.3f}")

    state = next_state

print(f"\n  Total reward (10 steps): {total_reward:.3f}")
print(f"  Average reward: {total_reward/10:.3f}")

# Test batch processing
print("\nTesting batch processing...")
batch_size = 5
states = jnp.tile(cheetah.reset(key), (batch_size, 1))
actions = jnp.tile(action, (batch_size, 1))
next_states = simplified_halfcheetah_step(states, actions)
print(f"  Batch states shape: {next_states.shape}")
print(f"  Expected shape: ({batch_size}, {cheetah.state_dim})")

print("\n" + "="*60)
print("Basic test completed successfully!")
print("="*60)

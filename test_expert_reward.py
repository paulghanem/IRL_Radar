#!/usr/bin/env python3
"""
Calculate expert reward for SimpleWalker2d
"""
import sys
sys.path.insert(0, 'src/control')
from simple_walker2d import simple_walker2d_reset, simple_walker2d_step, simple_walker2d_reward
import jax.numpy as jnp
import jax

# Simulate a simple forward trajectory
state = simple_walker2d_reset()
print('Initial state shape:', state.shape)
print('Initial state:', state[:6])

# Test 1: Zero action (standing still)
print("\n=== Test 1: Zero Action (Standing Still) ===")
state = simple_walker2d_reset()
total_reward = 0
for i in range(1000):
    action = jnp.zeros(6)
    next_state = simple_walker2d_step(state, action)
    reward = simple_walker2d_reward(state, action, next_state)
    total_reward += float(reward)
    state = next_state
    if i < 5:
        print(f"Step {i}: reward = {reward:.4f}, x_pos = {next_state[0]:.4f}")

print(f'Total reward (zero action, 1000 steps): {total_reward:.2f}')
print(f'Average reward per step: {total_reward / 1000:.4f}')

# Test 2: Small random actions
print("\n=== Test 2: Small Random Actions ===")
key = jax.random.PRNGKey(42)
state = simple_walker2d_reset()
total_reward = 0
for i in range(1000):
    key, subkey = jax.random.split(key)
    action = jax.random.normal(subkey, (6,)) * 0.1  # Small random actions
    next_state = simple_walker2d_step(state, action)
    reward = simple_walker2d_reward(state, action, next_state)
    total_reward += float(reward)
    state = next_state
    if i < 5:
        print(f"Step {i}: reward = {reward:.4f}, x_pos = {next_state[0]:.4f}")

print(f'Total reward (random actions, 1000 steps): {total_reward:.2f}')
print(f'Average reward per step: {total_reward / 1000:.4f}')

# Test 3: Forward action bias
print("\n=== Test 3: Forward Action Bias ===")
state = simple_walker2d_reset()
total_reward = 0
for i in range(1000):
    # Bias actions to encourage forward movement
    action = jnp.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
    next_state = simple_walker2d_step(state, action)
    reward = simple_walker2d_reward(state, action, next_state)
    total_reward += float(reward)
    state = next_state
    if i < 5:
        print(f"Step {i}: reward = {reward:.4f}, x_pos = {next_state[0]:.4f}")

print(f'Total reward (forward bias, 1000 steps): {total_reward:.2f}')
print(f'Average reward per step: {total_reward / 1000:.4f}')

print("\n=== Summary ===")
print("These rewards represent different policies on SimpleWalker2d")
print("Expert reward would come from a trained PPO agent")

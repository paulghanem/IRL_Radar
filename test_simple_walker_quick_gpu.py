#!/usr/bin/env python3
"""
Quick test of Simplified Walker2d on GPU
Reduced iterations for fast testing
"""
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import jax
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("QUICK GPU TEST - SIMPLIFIED WALKER2D")
print("=" * 80)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 80)

# Import simple Walker2d
from src.control.simple_walker2d import (
    simple_walker2d_step,
    simple_walker2d_reward,
    simple_walker2d_reset,
    get_simple_walker2d_params
)

# Test basic dynamics
print("\n1. Testing basic dynamics...")
state = simple_walker2d_reset()
action = jnp.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])

start = time.time()
for _ in range(10):  # Warmup
    state = simple_walker2d_step(state, action)
jax.block_until_ready(state)
warmup_time = time.time() - start
print(f"   Warmup (10 steps): {warmup_time:.4f}s")

state = simple_walker2d_reset()
start = time.time()
for _ in range(1000):
    state = simple_walker2d_step(state, action)
jax.block_until_ready(state)
elapsed = time.time() - start
print(f"   1000 steps: {elapsed:.4f}s ({elapsed/1000*1000:.2f}ms per step)")
print(f"   Final x position: {float(state[0]):.2f}")

# Test vectorized dynamics
print("\n2. Testing vectorized dynamics (GPU parallelism)...")
batch_size = 500
states = jnp.stack([simple_walker2d_reset() for _ in range(batch_size)])
actions = jnp.stack([action for _ in range(batch_size)])

# Warmup
start = time.time()
states_next = jax.vmap(simple_walker2d_step)(states, actions)
jax.block_until_ready(states_next)
warmup_time = time.time() - start
print(f"   Warmup ({batch_size} parallel): {warmup_time:.4f}s")

# Actual test
start = time.time()
for _ in range(100):
    states_next = jax.vmap(simple_walker2d_step)(states, actions)
jax.block_until_ready(states_next)
elapsed = time.time() - start
total_steps = batch_size * 100
print(f"   {total_steps} steps (100 iterations x {batch_size} parallel): {elapsed:.4f}s")
print(f"   Effective rate: {total_steps/elapsed:.0f} steps/second")

# Test with MPPI controller (quick version)
print("\n3. Testing with MPPI controller...")
from src.control.mppi_class import MPPI
from cost_jax import CostNN
from flax.training import train_state
import optax

env_params = get_simple_walker2d_params()
s_dim = env_params['state_dim']
a_dim = env_params['action_dim']

# Simple cost function
def cost_func(state, state_train):
    x_pos = state[:, 0]
    z_height = state[:, 1]
    body_angle = state[:, 2]

    forward_cost = -x_pos
    fall_penalty = jnp.where(
        (jnp.abs(body_angle) > 1.0) | (z_height < 0.8) | (z_height > 2.0),
        100.0,
        0.0
    )
    balance_cost = 0.5 * jnp.square(body_angle)

    costs = forward_cost + fall_penalty + balance_cost
    return costs.reshape(-1, 1)

# Dummy state train
class DummyStateTrain:
    def __init__(self):
        self.params = None
state_train_dummy = DummyStateTrain()

# Simple env wrapper
class SimpleWalker2dEnv:
    def reset(self, seed=None):
        return simple_walker2d_reset(), {}

def simple_dynamics(state, action):
    return simple_walker2d_step(state, action)

# Create MPPI with smaller horizon for quick test
mppi = MPPI(
    state_train=state_train_dummy,
    horizon=20,  # Reduced for quick test
    num_samples=100,  # Reduced for quick test
    dim_state=s_dim,
    dim_control=a_dim,
    dynamics=simple_dynamics,
    cost_func=cost_func,
    u_min=env_params['action_min'],
    u_max=env_params['action_max'],
    sigmas=jnp.ones(a_dim),
    lambda_=0.01,
    exploration=0.0,
    seed=123,
    env=SimpleWalker2dEnv(),
    mjx_model=None,
    gym_env='SimpleWalker2d',
    env_brax=None,
    use_mujoco=False
)

# Test MPPI forward pass
print("   Testing MPPI forward pass...")
state = simple_walker2d_reset()
prev_action_seq = jnp.zeros((20, a_dim))
key = jax.random.PRNGKey(123)

# Warmup
start = time.time()
action_seq, _, key, prev_action_seq = mppi.forward_pure(
    state=state,
    state_train=state_train_dummy,
    gail=False,
    key=key,
    prev_action_seq=prev_action_seq,
    frame_skip=5
)
jax.block_until_ready(action_seq)
warmup_time = time.time() - start
print(f"   Warmup: {warmup_time:.4f}s")

# Actual test
start = time.time()
for _ in range(10):
    action_seq, _, key, prev_action_seq = mppi.forward_pure(
        state=state,
        state_train=state_train_dummy,
        gail=False,
        key=key,
        prev_action_seq=prev_action_seq,
        frame_skip=5
    )
    jax.block_until_ready(action_seq)
elapsed = time.time() - start
print(f"   10 MPPI steps: {elapsed:.4f}s ({elapsed/10:.4f}s per step)")

# Quick rollout test
print("\n4. Quick rollout test (50 steps)...")
state = simple_walker2d_reset()
total_reward = 0.0

start = time.time()
for i in range(50):
    action_seq, _, key, prev_action_seq = mppi.forward_pure(
        state=state,
        state_train=state_train_dummy,
        gail=False,
        key=key,
        prev_action_seq=prev_action_seq,
        frame_skip=5
    )

    action = action_seq[0, :]
    next_state = simple_walker2d_step(state, action)
    reward = simple_walker2d_reward(state, action, next_state)

    total_reward += float(reward)
    state = next_state

jax.block_until_ready(state)
elapsed = time.time() - start

print(f"   Time: {elapsed:.4f}s ({elapsed/50:.4f}s per step)")
print(f"   Total reward: {total_reward:.2f}")
print(f"   Final x position: {float(state[0]):.2f}")

print("\n" + "=" * 80)
print("GPU TEST COMPLETE")
print("=" * 80)

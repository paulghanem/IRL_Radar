#!/usr/bin/env python3
"""
Test script to run trajectory generation with h=25, n=500 and print all array dtypes
"""

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import os
import argparse
import time

# Import the MPPI class
from src.control.mppi_class import MPPI
from src.reward_model.reward_model_nn import CostNN
from flax.training import train_state
import optax

print("=" * 70)
print("DTYPE TEST: h=25, num_traj=500")
print("=" * 70)
print()

# Check JAX configuration
print("JAX Configuration:")
print(f"  JAX version: {jax.__version__}")
print(f"  jax.config.x64_enabled: {jax.config.x64_enabled}")
print(f"  JAX devices: {jax.devices()}")
print()

# Setup arguments
class Args:
    def __init__(self):
        self.gym_env = "Walker2d"
        self.horizon = 25
        self.num_traj = 500
        self.N_steps = 100
        self.lr = 1e-4
        self.reward_fn_updates = 15
        self.lambda_ = 0.01
        self.rirl_iterations = 1
        self.UB = True
        self.save_images = False
        self.s_dim = 17
        self.a_dim = 6
        self.hidden_dim = 256
        self.seed = 42
        self.frame_skip = 4
        self.dt = 0.002

args = Args()

# Load MuJoCo model
assets_dir = "assets"
env_xml = "walker2d.xml"
model_path = os.path.join(assets_dir, env_xml)
model = mujoco.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(model)

print("Environment Setup:")
print(f"  Environment: {args.gym_env}")
print(f"  Horizon: {args.horizon}")
print(f"  Num trajectories: {args.num_traj}")
print(f"  N_steps: {args.N_steps}")
print()

# Initialize cost network
cost_f = CostNN(state_dims=args.s_dim, hidden_dim=args.hidden_dim)
key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key)
dummy_state = jnp.zeros((1, args.s_dim))
params = cost_f.init(subkey, dummy_state)
tx = optax.adam(args.lr)
state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)

# Initialize MPPI policy
policy = MPPI(
    state_train=state_train,
    action_dim=args.a_dim,
    horizon=args.horizon,
    num_samples=args.num_traj,
    mjx_model=mjx_model,
    gym_env=args.gym_env,
    args=args
)

print("=" * 70)
print("GENERATING TRAJECTORIES...")
print("=" * 70)
print()

# Load demo data (dummy for this test)
D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))

# Run trajectory generation
start_time = time.time()
trajs = [policy.generate_session_lax(args, state_train, D_demo)]
end_time = time.time()

print()
print("=" * 70)
print("EXECUTION COMPLETE")
print("=" * 70)
print(f"Execution time: {end_time - start_time:.4f} seconds")
print()

# Extract trajectory data
states, actions, total_rew, probs = trajs[0]

print("=" * 70)
print("ARRAY DTYPES:")
print("=" * 70)
print()

print("Trajectory outputs:")
print(f"  states.dtype:      {states.dtype}")
print(f"  states.shape:      {states.shape}")
print(f"  actions.dtype:     {actions.dtype}")
print(f"  actions.shape:     {actions.shape}")
print(f"  total_rew.dtype:   {total_rew.dtype}")
print(f"  total_rew.shape:   {total_rew.shape}")
print(f"  probs.dtype:       {probs.dtype}")
print(f"  probs.shape:       {probs.shape}")
print()

# Check internal MPPI arrays
print("MPPI internal arrays:")
print(f"  mjx_model dtype: {type(mjx_model)}")
print(f"  mjx_data qpos dtype: {policy.mjx_data.qpos.dtype}")
print(f"  mjx_data qvel dtype: {policy.mjx_data.qvel.dtype}")
print()

# Test explicit float64 request
print("Testing explicit dtype requests:")
test_arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print(f"  Requested: jnp.float64")
print(f"  Actual:    {test_arr.dtype}")
print()

print("=" * 70)
print("SUMMARY:")
print("=" * 70)
if states.dtype == jnp.float32:
    print("  ✓ Trajectories are computed in FLOAT32 (FP32)")
elif states.dtype == jnp.float64:
    print("  ✓ Trajectories are computed in FLOAT64 (FP64)")
else:
    print(f"  ? Unexpected dtype: {states.dtype}")
print("=" * 70)

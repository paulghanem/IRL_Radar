# Debug script to check Walker2d dimensions
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import gymnasium as gym
from mujoco import mjx

env = gym.make("Walker2d-v4")
state, info = env.reset(seed=123)

mjx_model = mjx.put_model(env.unwrapped.model)
mjx_data = mjx.put_data(env.unwrapped.model, env.unwrapped.data)

print("=== Walker2d-v4 Dimensions ===")
print(f"Observation space shape: {env.observation_space.shape[0]}")
print(f"env.unwrapped.model.nq (position DOFs): {env.unwrapped.model.nq}")
print(f"env.unwrapped.model.nv (velocity DOFs): {env.unwrapped.model.nv}")
print(f"mjx_model.nq: {mjx_model.nq}")
print(f"mjx_model.nv: {mjx_model.nv}")
print(f"mjx_data.qpos shape: {mjx_data.qpos.shape}")
print(f"mjx_data.qvel shape: {mjx_data.qvel.shape}")
print(f"State from env.reset() shape: {state.shape}")
print(f"nq + nv = {mjx_model.nq + mjx_model.nv}")
print("")
print(f"mjx_data.qpos: {mjx_data.qpos}")
print(f"mjx_data.qvel: {mjx_data.qvel}")
print(f"State from env: {state}")

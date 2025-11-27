#!/usr/bin/env python3
"""Quick 10-timestep GPU test for Walker2d"""
import os
import sys
import glob

# Configure JAX for GPU - must be done BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Find all nvidia CUDA library paths from site-packages
print("Configuring CUDA library paths...")
site_packages = None
for path in sys.path:
    if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
        site_packages = path
        break

if site_packages:
    nvidia_path = os.path.join(site_packages, 'nvidia')
    subdirs = [d for d in os.listdir(nvidia_path) if os.path.isdir(os.path.join(nvidia_path, d))]

    lib_paths = []
    for subdir in subdirs:
        lib_path = os.path.join(nvidia_path, subdir, 'lib')
        if os.path.exists(lib_path):
            lib_paths.append(lib_path)

    if lib_paths:
        # Set LD_LIBRARY_PATH to include all nvidia library paths
        existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(lib_paths)
        if existing_ld_path:
            new_ld_path = f"{new_ld_path}:{existing_ld_path}"
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"Added {len(lib_paths)} nvidia library paths to LD_LIBRARY_PATH")
        print(f"Site-packages nvidia location: {nvidia_path}")
    else:
        print("WARNING: No nvidia library paths found in site-packages")
else:
    print("WARNING: site-packages nvidia directory not found")
import jax
import jax.numpy as jnp
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

print("=" * 60)
print("QUICK 10-TIMESTEP GPU TEST")
print("=" * 60)

# Check JAX devices
print(f"JAX devices: {jax.devices()}")
print(f"JAX version: {jax.__version__}")

# Test expert model loading
print("\n1. Loading Walker2d expert model...")
env = gym.make("Walker2d-v4", exclude_current_positions_from_observation=False)
model = PPO("MlpPolicy", env)
model = model.load("expert_agents/Walker2d/PPO.zip", env)
print("✓ Expert model loaded successfully")

# Generate 10 timesteps
print("\n2. Generating 10 expert timesteps...")
vec_env = model.get_env()
vec_env._seeds = [123]
obs = vec_env.reset()

states = []
actions = []
rewards = []

for t in range(10):
    states.append(obs.ravel())
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    actions.append(action.ravel())
    rewards.append(reward)
    print(f"  t={t}: reward={reward[0]:.3f}, cumsum={np.sum(rewards):.3f}")

print(f"\n✓ Generated {len(states)} timesteps")
print(f"✓ Total reward: {np.sum(rewards):.2f}")

print("\n3. Testing JAX operations...")
test_array = jnp.array(states)
print(f"✓ JAX array shape: {test_array.shape}")
print(f"✓ JAX array mean: {jnp.mean(test_array):.6f}")

print("\n" + "=" * 60)
print("QUICK TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)

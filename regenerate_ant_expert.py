#!/usr/bin/env python3
"""
Regenerate Ant expert using Stable Baselines3 PPO
This ensures compatibility with current NumPy version
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np
import torch

print("=" * 60)
print("Regenerating Ant Expert with Stable Baselines3")
print("=" * 60)
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Create environment - NOW INCLUDING x-position
env = gym.make("Ant-v4", exclude_current_positions_from_observation=False)

# Create PPO model with GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

print("\nCreating PPO model...")
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Create checkpoint callback - save every 200k timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="expert_agents/Ant/",
    name_prefix="PPO",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# Train the model
print("\nTraining for 10M timesteps (this will take a while)...")
print("Checkpoints will be saved every 200k timesteps to expert_agents/Ant/")
model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

# Save the model
save_path = "expert_agents/Ant/PPO.zip"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

print(f"\n✓ Expert saved to {save_path}")
print("=" * 60)

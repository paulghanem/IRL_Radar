#!/usr/bin/env python3
"""
Improved HalfCheetah Expert Training with Optimized Hyperparameters
Based on Stable-Baselines3 best practices and research benchmarks
"""
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch

print("=" * 70)
print("HalfCheetah-v4 Expert Training (Improved Hyperparameters)")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Environment setup
env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)
eval_env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}\n")

# Improved PPO hyperparameters based on RL Baselines3 Zoo
# These are tuned for HalfCheetah performance
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,           # Standard learning rate
    n_steps=512,                  # Reduced from default 2048 for more frequent updates
    batch_size=64,                # Smaller batches for more gradient updates
    n_epochs=20,                  # More epochs per update
    gamma=0.98,                   # Slightly reduced discount for faster learning
    gae_lambda=0.92,              # GAE parameter for advantage estimation
    clip_range=0.2,               # Standard PPO clip range
    ent_coef=0.0,                 # No entropy bonus (deterministic policy)
    vf_coef=0.5,                  # Value function coefficient
    max_grad_norm=0.5,            # Gradient clipping
    verbose=1,
    device=device,
    tensorboard_log="./logs_halfcheetah_improved/"
)

# Checkpoint callback - save every 200k timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="expert_agents/HalfCheetah-v4/",
    name_prefix="PPO_improved",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# Evaluation callback - evaluate every 100k steps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="expert_agents/HalfCheetah-v4/",
    log_path="./logs_halfcheetah_improved/",
    eval_freq=100_000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
)

# Train for 10M timesteps (longer than current 6.6M)
print("Starting training for 10M timesteps...")
print("Target performance: 4000-6000+ reward per episode")
print("Checkpoints will be saved every 200k timesteps")
print("=" * 70)
print()

model.learn(
    total_timesteps=10_000_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Save final model
save_path = "expert_agents/HalfCheetah-v4/PPO_improved_final.zip"
model.save(save_path)
print(f"\n✓ Final model saved to {save_path}")

# Test the trained agent
print("\nTesting final model for 1000 steps...")
obs, _ = env.reset()
total_reward = 0
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

env.close()
eval_env.close()

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"Final test reward (1000 steps): {total_reward:.2f}")
print(f"{'='*70}")

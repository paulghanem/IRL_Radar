# -*- coding: utf-8 -*-
"""
Correct PPO training script for HalfCheetah-v4
@author: paul
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# -----------------------------
# Config
# -----------------------------
env_name = "HalfCheetah-v4"
model_dir = "expert_agents"
log_dir = "logs_HalfCheetah-v4"
total_training_steps = 10_000_000   # 10 million
checkpoint_every = 1_000_000        # 1M-step checkpoints

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# -----------------------------
# Create environment (vectorized + normalized)
# -----------------------------
def make_env():
    return gym.make(env_name, exclude_current_positions_from_observation=False)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# -----------------------------
# PPO hyperparameters tuned for MuJoCo
# -----------------------------
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    verbose=1,
    tensorboard_log=log_dir,
)

# -----------------------------
# Checkpoint callback
# -----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint_every // env.num_envs,
    save_path=model_dir,
    name_prefix="halfcheetah_ppo",
)

# -----------------------------
# Train PPO
# -----------------------------
model.learn(
    total_timesteps=total_training_steps,
    callback=checkpoint_callback,
)

# Save final model + normalization stats
model.save(f"{model_dir}/PPO_HalfCheetah_Final")
env.save(f"{model_dir}/vecnormalize_stats.pkl")

# -----------------------------
# Evaluate trained policy
# -----------------------------
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load(f"{model_dir}/vecnormalize_stats.pkl", eval_env)
eval_env.training = False  # Important: turn off normalization updates

obs = eval_env.reset()
total_reward = 0.0

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward[0]
    if done:
        obs = eval_env.reset()

print("Evaluation reward over 1000 steps:", total_reward)

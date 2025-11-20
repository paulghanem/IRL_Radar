# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:01:40 2025
@author: siliconsynapse
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO

model_dir = "expert_agents"
log_dir = "logs_HalfCheetah-v4"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

sb3_algo = "PPO"
TIMESTEPS = 10000
env_name = "HalfCheetah-v4"

env = gym.make(env_name, exclude_current_positions_from_observation=False)

train = True
load = False

if train:
    iterations = 0
    # ⭐ BEST-KNOWN HalfCheetah PPO hyperparameters ⭐
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        # === Stable-Baselines3 RL-Zoo best values === #
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,            # ⭐ helps a lot for HalfCheetah
        sde_sample_freq=4,

        tensorboard_log=log_dir,
        device="cuda"            # IMPORTANT: forces GPU usage
    )

elif load:
    iterations = 0
    model = PPO.load(f"{model_dir}/{env_name}/{sb3_algo}")
    model.set_env(env)

# Train PPO agent
while iterations < 1000:
    iterations += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/{env_name}_1/{sb3_algo}_{TIMESTEPS * iterations}")

model.save(f"{model_dir}/{env_name}/{sb3_algo}")

# Test the trained agent
obs, _ = env.reset()
for _ in range(TIMESTEPS):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

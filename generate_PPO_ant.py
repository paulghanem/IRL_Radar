# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:01:40 2025
@author: siliconsynapse
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO

# ============================================================
# Create a NEW folder for this run inside expert_agents/Ant
# ============================================================
RUN_NAME = "run_01"   # change for each experiment: run_02, run_03 ...

model_dir = f"expert_agents/Ant/{RUN_NAME}/models"
log_dir   = f"expert_agents/Ant/{RUN_NAME}/logs"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

sb3_algo = "PPO"
TIMESTEPS = 10000
env_name = "Ant-v4"   # <--- IMPORTANT! Must include version

# ============================================================
# Environment
# ============================================================
env = gym.make(env_name, exclude_current_positions_from_observation=False)

train = True
load = False

# ============================================================
# PPO Hyperparameters (from SB3 RL-Zoo tuned for Ant)
# ============================================================
if train:
    iterations = 0

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        # === PPO hyperparameters optimized for Ant ===
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,

        use_sde=False,     # For Ant, SDE = BAD (makes unstable gait)

        tensorboard_log=log_dir,
        device="cuda"      # force GPU
    )

elif load:
    iterations = 0
    model = PPO.load(f"{model_dir}/{sb3_algo}_final")
    model.set_env(env)

# ============================================================
# Training Loop
# ============================================================
while iterations < 1000:
    iterations += 1
    print(f"\n=== PPO Training Iteration {iterations} ===")

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    # Save checkpoint
    checkpoint_path = f"{model_dir}/{sb3_algo}_{TIMESTEPS * iterations}"
    print(f"Saving checkpoint: {checkpoint_path}")
    model.save(checkpoint_path)

# Final save
final_model_path = f"{model_dir}/{sb3_algo}_final"
model.save(final_model_path)
print(f"Final model saved: {final_model_path}")

# ============================================================
# Test the trained agent
# ============================================================
obs, _ = env.reset()
total_reward = 0

for _ in range(TIMESTEPS):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

env.close()

print("===============================================")
print("Test episode reward:", total_reward)
print("===============================================")

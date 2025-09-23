# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:01:40 2025

@author: siliconsynapse
"""
import os
import gymnasium as gym
from stable_baselines3 import PPO,SAC

model_dir = "expert_agents"
log_dir = "logs_humanoid_standup"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
sb3_algo="PPO"
TIMESTEPS = 200
env_name="MountainCarContinuous-v0"
# Create environment (you can also try "Ant-v4", "Humanoid-v4", etc.)
#env = gym.make(env_name,exclude_current_positions_from_observation=False,render_mode="human")
env = gym.make(env_name,render_mode="human")
#env = gym.make(env_name, render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)


train=True
load=False
if train==True:
    iterations=0
    model = PPO("MlpPolicy", env, verbose=1)
    #model = PPO("CnnPolicy", env, verbose=1)
elif load:
    iterations=299
    model = PPO.load((f"{model_dir}/{env_name}/{sb3_algo}_{TIMESTEPS * iterations}"))
    
# Train PPO agent


while iterations<300:
    iterations+=1
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False)
    #model.save(f"{model_dir}/{env_name}/{sb3_algo}_{TIMESTEPS * iterations}")
#model.save(f"{model_dir}/{env_name}/{sb3_algo}")
# Test the trained agent
obs, _ = env.reset()
for _ in range(TIMESTEPS):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    

    if terminated or truncated:
        obs, _ = env.reset()

env.close()




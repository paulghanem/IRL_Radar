import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load the latest Walker2d expert checkpoint
model_path = "expert_agents/Walker2d/PPO_6400000_steps.zip"
model = PPO.load(model_path)

# Create environment (with x-position included, same as training)
env = gym.make("Walker2d-v4", exclude_current_positions_from_observation=False)

# Reset environment
obs, info = env.reset()

# Run for 1000 steps
num_steps = 1000
cumulative_reward = 0.0
done = False

print(f"Testing Walker2d expert (6.4M steps checkpoint) for {num_steps} rollout steps...\n")

for step in range(num_steps):
    # Get action from expert policy
    action, _ = model.predict(obs, deterministic=True)

    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    cumulative_reward += reward

    # Print progress every 100 steps
    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}/{num_steps}: Cumulative Reward = {cumulative_reward:.2f}")

    # Reset if episode ends
    if done:
        print(f"  Episode ended at step {step + 1}, resetting environment...")
        obs, info = env.reset()

env.close()

print(f"\n{'='*60}")
print(f"Test Complete!")
print(f"{'='*60}")
print(f"Total Steps: {num_steps}")
print(f"Cumulative Reward: {cumulative_reward:.2f}")
print(f"Average Reward per Step: {cumulative_reward/num_steps:.4f}")
print(f"{'='*60}")

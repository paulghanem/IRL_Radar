import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
import os

def test_expert_model(model_path, env_name, rollout_length=1000, num_episodes=1):
    """
    Test an expert model and return cumulative reward.

    Args:
        model_path: Path to the trained model zip file
        env_name: Name of the Gym environment
        rollout_length: Maximum steps per episode
        num_episodes: Number of episodes to test

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Testing {env_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load the trained model
    try:
        model = TD3.load(model_path)
        print(f"[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    # Create environment
    try:
        env = gym.make(env_name, render_mode=None)
        print(f"[OK] Environment created: {env_name}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return None

    results = {
        'env_name': env_name,
        'total_rewards': [],
        'episode_lengths': []
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        cumulative_reward = 0
        done = False
        truncated = False
        step = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not (done or truncated) and step < rollout_length:
            # Get action from expert policy
            action, _states = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            step += 1

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}/{rollout_length} | Cumulative Reward: {cumulative_reward:.2f}")

        results['total_rewards'].append(cumulative_reward)
        results['episode_lengths'].append(step)

        print(f"  Episode finished at step {step}")
        print(f"  Total Cumulative Reward: {cumulative_reward:.2f}")
        print(f"  Done: {done}, Truncated: {truncated}")

    env.close()

    # Calculate statistics
    results['mean_reward'] = np.mean(results['total_rewards'])
    results['std_reward'] = np.std(results['total_rewards'])
    results['mean_length'] = np.mean(results['episode_lengths'])

    return results


def main():
    # Expert models to test - using v4 environments which work with newer MuJoCo
    experts = {
        'HalfCheetah-v4': 'experts/td3-HalfCheetah-v3.zip',
        'Hopper-v4': 'experts/td3-Hopper-v3.zip',
        'Walker2d-v4': 'experts/td3-Walker2d-v3.zip',
        'Ant-v4': 'experts/td3-Ant-v3.zip'
    }

    rollout_length = 1000
    num_episodes = 1

    print("\n" + "="*60)
    print("EXPERT MODELS EVALUATION")
    print("="*60)
    print(f"Rollout Length: {rollout_length} steps")
    print(f"Number of Episodes: {num_episodes}")
    print("="*60)

    all_results = {}

    # Test each expert model
    for env_name, model_path in experts.items():
        if not os.path.exists(model_path):
            print(f"\n[WARNING] Model not found: {model_path}")
            continue

        results = test_expert_model(model_path, env_name, rollout_length, num_episodes)

        if results:
            all_results[env_name] = results

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Environment':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Length':<15}")
    print("-"*60)

    for env_name, results in all_results.items():
        print(f"{env_name:<20} {results['mean_reward']:<15.2f} {results['std_reward']:<15.2f} {results['mean_length']:<15.1f}")

    print("="*60)

    return all_results


if __name__ == "__main__":
    results = main()

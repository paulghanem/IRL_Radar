import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
import os
import warnings
warnings.filterwarnings('ignore')

def test_expert_model(model_path, env_name, algo_class, rollout_length=1000):
    """
    Test an expert model and return cumulative reward.

    Args:
        model_path: Path to the trained model zip file
        env_name: Name of the Gym environment
        algo_class: Algorithm class (TD3, PPO, or SAC)
        rollout_length: Maximum steps per episode

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Testing {env_name}")
    print(f"Model: {model_path}")
    print(f"Algorithm: {algo_class.__name__}")
    print(f"{'='*60}")

    # Load the trained model
    try:
        model = algo_class.load(model_path)
        print(f"[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    # Create environment
    try:
        env = gym.make(env_name, render_mode=None)
        print(f"[OK] Environment created: {env_name}")
        print(f"Observation space: {env.observation_space.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return None

    results = {
        'env_name': env_name,
        'model_path': model_path,
        'algorithm': algo_class.__name__,
        'total_reward': 0,
        'episode_length': 0
    }

    try:
        obs, info = env.reset()
        cumulative_reward = 0
        done = False
        truncated = False
        step = 0

        print(f"\nStarting rollout...")

        while not (done or truncated) and step < rollout_length:
            # Get action from expert policy
            action, _states = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            step += 1

            # Print progress every 200 steps
            if step % 200 == 0:
                print(f"  Step {step}/{rollout_length} | Cumulative Reward: {cumulative_reward:.2f}")

        results['total_reward'] = cumulative_reward
        results['episode_length'] = step
        results['avg_reward'] = cumulative_reward / step if step > 0 else 0

        print(f"\nEpisode finished at step {step}")
        print(f"Total Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Average Reward per Step: {results['avg_reward']:.2f}")
        print(f"Done: {done}, Truncated: {truncated}")

    except Exception as e:
        print(f"[ERROR] Error during rollout: {e}")
        results['error'] = str(e)
        return None
    finally:
        env.close()

    return results


def main():
    # Expert models to test with their corresponding algorithms and environments
    experts = [
        # Original working models
        ('experts/td3-HalfCheetah-v3.zip', 'HalfCheetah-v4', TD3),
        ('experts/td3-Hopper-v3.zip', 'Hopper-v4', TD3),
        ('experts/td3-Walker2d-v3.zip', 'Walker2d-v4', TD3),

        # New Ant models (multiple algorithms)
        ('experts/ppo-Ant-v4.zip', 'Ant-v4', PPO),
        ('experts/sac-Ant-v4.zip', 'Ant-v4', SAC),
        ('experts/td3-Ant-v4.zip', 'Ant-v4', TD3),

        # Swimmer models
        ('experts/ppo-Swimmer-v4.zip', 'Swimmer-v4', PPO),
        ('experts/sac-Swimmer-v4.zip', 'Swimmer-v4', SAC),
        ('experts/td3-Swimmer-v4.zip', 'Swimmer-v4', TD3),

        # Humanoid model
        ('experts/sac-Humanoid-v4.zip', 'Humanoid-v4', SAC),
    ]

    rollout_length = 1000

    print("\n" + "="*60)
    print("COMPREHENSIVE EXPERT MODELS EVALUATION")
    print("="*60)
    print(f"Rollout Length: {rollout_length} steps")
    print("="*60)

    all_results = []

    # Test each expert model
    for model_path, env_name, algo_class in experts:
        if not os.path.exists(model_path):
            print(f"\n[WARNING] Model not found: {model_path}")
            continue

        results = test_expert_model(model_path, env_name, algo_class, rollout_length)

        if results:
            all_results.append(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Environment':<20} {'Algorithm':<10} {'Total Reward':<15} {'Avg/Step':<10}")
    print("-"*60)

    for results in all_results:
        env_name = results['env_name']
        algo = results['algorithm']
        total = results['total_reward']
        avg = results['avg_reward']
        print(f"{env_name:<20} {algo:<10} {total:<15.2f} {avg:<10.2f}")

    print("="*60)

    # Group by environment
    print("\n" + "="*60)
    print("BEST MODELS BY ENVIRONMENT")
    print("="*60)

    by_env = {}
    for results in all_results:
        env = results['env_name']
        if env not in by_env:
            by_env[env] = []
        by_env[env].append(results)

    for env_name, env_results in sorted(by_env.items()):
        best = max(env_results, key=lambda x: x['total_reward'])
        print(f"\n{env_name}:")
        print(f"  Best Algorithm: {best['algorithm']}")
        print(f"  Total Reward: {best['total_reward']:.2f}")
        print(f"  Model: {os.path.basename(best['model_path'])}")

    print("="*60)

    return all_results


if __name__ == "__main__":
    results = main()

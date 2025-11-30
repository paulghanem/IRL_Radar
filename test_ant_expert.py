import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
import warnings
warnings.filterwarnings('ignore')


class ObservationPaddingWrapper(gym.ObservationWrapper):
    """Wrapper to pad observations to match expected dimensions."""
    def __init__(self, env, target_shape):
        super().__init__(env)
        self.target_shape = target_shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=target_shape,
            dtype=np.float64
        )

    def observation(self, obs):
        if len(obs) < self.target_shape[0]:
            # Pad with zeros
            padded = np.zeros(self.target_shape, dtype=np.float64)
            padded[:len(obs)] = obs
            return padded
        elif len(obs) > self.target_shape[0]:
            # Truncate
            return obs[:self.target_shape[0]]
        return obs

def test_ant_expert(rollout_length=1000):
    """
    Test the Ant-v3 expert model.
    """
    model_path = 'experts/td3-Ant-v3.zip'
    env_name = 'Ant-v5'  # Try v5 first, then v4, then v3

    print(f"\n{'='*60}")
    print(f"Testing Ant Expert Model")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load the trained model
    try:
        model = TD3.load(model_path)
        print(f"[OK] Model loaded successfully")
        print(f"Model observation space: {model.observation_space}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    # Try different environment versions with compatibility wrapper
    env = None

    # First try with exclude_current_positions_from_observation=False for v4/v5
    for version in ['v4', 'v5']:
        try:
            test_env_name = f'Ant-{version}'
            test_env = gym.make(
                test_env_name,
                render_mode=None,
                exclude_current_positions_from_observation=False  # Include position to match v3
            )
            print(f"[OK] Environment created: {test_env_name} (with position)")
            print(f"Environment observation space: {test_env.observation_space}")

            # Check if observation spaces match or can be wrapped
            if test_env.observation_space.shape == model.observation_space.shape:
                print(f"[OK] Observation spaces match!")
                env = test_env
                env_name = test_env_name
                break
            else:
                print(f"[WARNING] Observation space mismatch. Shape: {test_env.observation_space.shape} vs {model.observation_space.shape}")
                # Try wrapping with padding if close enough
                if abs(test_env.observation_space.shape[0] - model.observation_space.shape[0]) <= 10:
                    print(f"[INFO] Attempting to use padding wrapper...")
                    wrapped_env = ObservationPaddingWrapper(test_env, model.observation_space.shape)
                    print(f"[OK] Wrapped observation space: {wrapped_env.observation_space}")
                    env = wrapped_env
                    env_name = f"{test_env_name} (padded)"
                    break
                else:
                    test_env.close()
        except Exception as e:
            print(f"[WARNING] Failed to create {test_env_name}: {e}")
            continue

    if env is None:
        print(f"[ERROR] Could not find compatible environment version")
        return None

    print(f"\nUsing environment: {env_name}")
    print(f"Starting {rollout_length}-step rollout...")
    print(f"{'='*60}\n")

    # Run episode
    obs, info = env.reset()
    cumulative_reward = 0
    done = False
    truncated = False
    step = 0

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

    env.close()

    print(f"\n{'='*60}")
    print(f"Episode finished at step {step}")
    print(f"Total Cumulative Reward: {cumulative_reward:.2f}")
    print(f"Average Reward per Step: {cumulative_reward/step:.2f}")
    print(f"Done: {done}, Truncated: {truncated}")
    print(f"{'='*60}")

    return {
        'env_name': env_name,
        'total_reward': cumulative_reward,
        'episode_length': step,
        'avg_reward': cumulative_reward / step
    }


if __name__ == "__main__":
    results = test_ant_expert(rollout_length=1000)

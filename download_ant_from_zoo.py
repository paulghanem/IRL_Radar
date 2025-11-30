"""
Try to download Ant model using rl_zoo3 package or direct GitHub access.
"""
import os
import urllib.request
import zipfile

print("="*60)
print("Attempting to download Ant expert from RL Zoo")
print("="*60)

# Try direct download from RL Baselines3 Zoo GitHub repository
# These are the actual trained models from the zoo
github_models = [
    {
        'name': 'ant-ppo-from-zoo-repo',
        'url': 'https://github.com/DLR-RM/rl-baselines3-zoo/raw/master/trained_agents/ppo/Ant-v3/Ant-v3.zip',
        'output': 'experts/ppo-Ant-v3-zoo.zip',
        'env': 'Ant-v3'
    },
    {
        'name': 'ant-sac-from-zoo-repo',
        'url': 'https://github.com/DLR-RM/rl-baselines3-zoo/raw/master/trained_agents/sac/Ant-v3/Ant-v3.zip',
        'output': 'experts/sac-Ant-v3-zoo.zip',
        'env': 'Ant-v3'
    },
    {
        'name': 'ant-td3-from-zoo-repo',
        'url': 'https://github.com/DLR-RM/rl-baselines3-zoo/raw/master/trained_agents/td3/Ant-v3/Ant-v3.zip',
        'output': 'experts/td3-Ant-v3-zoo.zip',
        'env': 'Ant-v3'
    },
]

os.makedirs('experts', exist_ok=True)

downloaded = []

for model_info in github_models:
    name = model_info['name']
    url = model_info['url']
    output = model_info['output']

    print(f"\nTrying: {name}")
    print(f"URL: {url}")

    try:
        urllib.request.urlretrieve(url, output)
        file_size = os.path.getsize(output) / (1024 * 1024)
        print(f"[SUCCESS] Downloaded! Size: {file_size:.2f} MB")
        downloaded.append((output, model_info['env']))

    except Exception as e:
        print(f"[FAILED] {e}")
        if os.path.exists(output):
            os.remove(output)

print("\n" + "="*60)
print(f"Downloaded {len(downloaded)} models")
print("="*60)

# Now test them to see if any are compatible
if downloaded:
    print("\nNow testing compatibility...")

    import gymnasium as gym
    from stable_baselines3 import PPO, SAC, TD3

    for model_path, env_name in downloaded:
        print(f"\n{'='*60}")
        print(f"Testing: {model_path}")
        print(f"{'='*60}")

        # Determine algorithm from filename
        if 'ppo' in model_path.lower():
            algo_class = PPO
        elif 'sac' in model_path.lower():
            algo_class = SAC
        elif 'td3' in model_path.lower():
            algo_class = TD3
        else:
            continue

        try:
            model = algo_class.load(model_path)
            print(f"Model loaded successfully")
            print(f"Expected observation space: {model.observation_space.shape}")

            # Try with v4
            try:
                env = gym.make('Ant-v4')
                print(f"Ant-v4 observation space: {env.observation_space.shape}")

                if env.observation_space.shape == model.observation_space.shape:
                    print("✓ COMPATIBLE WITH Ant-v4!")
                else:
                    print(f"✗ Incompatible (need {model.observation_space.shape}, got {env.observation_space.shape})")

                env.close()
            except Exception as e:
                print(f"Error with v4: {e}")

            # Try with v5
            try:
                env = gym.make('Ant-v5')
                print(f"Ant-v5 observation space: {env.observation_space.shape}")

                if env.observation_space.shape == model.observation_space.shape:
                    print("✓ COMPATIBLE WITH Ant-v5!")
                else:
                    print(f"✗ Incompatible (need {model.observation_space.shape}, got {env.observation_space.shape})")

                env.close()
            except Exception as e:
                print(f"Error with v5: {e}")

        except Exception as e:
            print(f"Error loading model: {e}")

print("\n" + "="*60)
print("Search complete")
print("="*60)

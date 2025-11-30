import urllib.request
import os

# Create directory for experts
os.makedirs('experts', exist_ok=True)

# Model URLs from Hugging Face - Using newer trained models
# These are trained on more recent versions and should be compatible
models = {
    # Try v4 models which should be more compatible with current Gymnasium
    'Ant-v4': 'https://huggingface.co/sb3/ppo-Ant-v3/resolve/main/ppo-Ant-v3.zip',
    'Swimmer-v4': 'https://huggingface.co/sb3/ppo-Swimmer-v3/resolve/main/ppo-Swimmer-v3.zip',
    'Humanoid-v4': 'https://huggingface.co/sb3/ppo-Humanoid-v3/resolve/main/ppo-Humanoid-v3.zip',

    # Alternative TD3 models
    'Ant-v4-td3': 'https://huggingface.co/sb3/td3-Ant-v3/resolve/main/td3-Ant-v3.zip',
    'Swimmer-v4-td3': 'https://huggingface.co/sb3/td3-Swimmer-v3/resolve/main/td3-Swimmer-v3.zip',

    # Try SAC models as well
    'Ant-v4-sac': 'https://huggingface.co/sb3/sac-Ant-v3/resolve/main/sac-Ant-v3.zip',
    'Swimmer-v4-sac': 'https://huggingface.co/sb3/sac-Swimmer-v3/resolve/main/sac-Swimmer-v3.zip',
    'Humanoid-v4-sac': 'https://huggingface.co/sb3/sac-Humanoid-v3/resolve/main/sac-Humanoid-v3.zip',
}

print("Downloading additional expert models from Hugging Face...")
print("=" * 60)

downloaded = []
failed = []

for env_name, url in models.items():
    algo = 'ppo'
    if '-td3' in env_name:
        algo = 'td3'
    elif '-sac' in env_name:
        algo = 'sac'

    base_name = env_name.replace('-td3', '').replace('-sac', '')
    output_file = f'experts/{algo}-{base_name}.zip'

    # Skip if already exists
    if os.path.exists(output_file):
        print(f"\n[SKIP] {output_file} already exists")
        continue

    print(f"\nDownloading {env_name} ({algo.upper()})...")
    print(f"URL: {url}")
    print(f"Saving to: {output_file}")

    try:
        urllib.request.urlretrieve(url, output_file)
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
        print(f"[OK] Downloaded successfully! Size: {file_size:.2f} MB")
        downloaded.append(output_file)
    except Exception as e:
        print(f"[ERROR] Error downloading {env_name}: {e}")
        failed.append(env_name)
        # Remove partial download
        if os.path.exists(output_file):
            os.remove(output_file)

print("\n" + "=" * 60)
print("Download Summary:")
print(f"Successfully downloaded: {len(downloaded)} models")
for model in downloaded:
    print(f"  ✓ {model}")

if failed:
    print(f"\nFailed to download: {len(failed)} models")
    for model in failed:
        print(f"  ✗ {model}")

print("=" * 60)

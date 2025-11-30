import urllib.request
import os
import json

# Try multiple sources for Ant models compatible with v4/v5

models_to_try = [
    # Try RL Baselines3 Zoo trained models (community uploads)
    {
        'name': 'ant-v4-ppo-rlzoo',
        'url': 'https://huggingface.co/sb3/ppo-Ant-v4/resolve/main/ppo-Ant-v4.zip',
        'output': 'experts/ppo-Ant-v4-new.zip'
    },
    {
        'name': 'ant-v4-sac-rlzoo',
        'url': 'https://huggingface.co/sb3/sac-Ant-v4/resolve/main/sac-Ant-v4.zip',
        'output': 'experts/sac-Ant-v4-new.zip'
    },
    {
        'name': 'ant-v4-td3-rlzoo',
        'url': 'https://huggingface.co/sb3/td3-Ant-v4/resolve/main/td3-Ant-v4.zip',
        'output': 'experts/td3-Ant-v4-new.zip'
    },
    # Try cleanrl models
    {
        'name': 'ant-cleanrl-ppo',
        'url': 'https://huggingface.co/cleanrl/ppo-Ant-v4/resolve/main/ppo-Ant-v4.zip',
        'output': 'experts/ppo-Ant-v4-cleanrl.zip'
    },
    # Try newer gymnasium-based models
    {
        'name': 'ant-v5-ppo',
        'url': 'https://huggingface.co/sb3/ppo-Ant-v5/resolve/main/ppo-Ant-v5.zip',
        'output': 'experts/ppo-Ant-v5.zip'
    },
    {
        'name': 'ant-v2-td3',
        'url': 'https://huggingface.co/sb3/td3-Ant-v2/resolve/main/td3-Ant-v2.zip',
        'output': 'experts/td3-Ant-v2.zip'
    },
]

print("="*60)
print("Searching for compatible Ant models...")
print("="*60)

os.makedirs('experts', exist_ok=True)

downloaded = []
failed = []

for model_info in models_to_try:
    name = model_info['name']
    url = model_info['url']
    output = model_info['output']

    if os.path.exists(output):
        print(f"\n[SKIP] {output} already exists")
        downloaded.append(output)
        continue

    print(f"\nTrying: {name}")
    print(f"URL: {url}")
    print(f"Output: {output}")

    try:
        urllib.request.urlretrieve(url, output)
        file_size = os.path.getsize(output) / (1024 * 1024)
        print(f"[SUCCESS] Downloaded! Size: {file_size:.2f} MB")
        downloaded.append(output)
    except urllib.error.HTTPError as e:
        print(f"[FAILED] HTTP {e.code}: {e.reason}")
        failed.append(name)
        if os.path.exists(output):
            os.remove(output)
    except Exception as e:
        print(f"[FAILED] {e}")
        failed.append(name)
        if os.path.exists(output):
            os.remove(output)

print("\n" + "="*60)
print("Download Summary:")
print("="*60)
print(f"Successfully downloaded: {len(downloaded)}")
for model in downloaded:
    print(f"  + {model}")

if failed:
    print(f"\nFailed: {len(failed)}")
    for name in failed:
        print(f"  - {name}")

print("="*60)

# Now test what we downloaded
if downloaded:
    print("\nProceeding to test downloaded models...")
else:
    print("\nNo models downloaded. You may need to train your own Ant expert.")

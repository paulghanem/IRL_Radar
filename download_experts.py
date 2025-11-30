import urllib.request
import os

# Create directory for experts
os.makedirs('experts', exist_ok=True)

# Model URLs from Hugging Face
models = {
    'HalfCheetah-v3': 'https://huggingface.co/sb3/td3-HalfCheetah-v3/resolve/main/td3-HalfCheetah-v3.zip',
    'Hopper-v3': 'https://huggingface.co/sb3/td3-Hopper-v3/resolve/main/td3-Hopper-v3.zip',
    'Walker2d-v3': 'https://huggingface.co/sb3/td3-Walker2d-v3/resolve/main/td3-Walker2d-v3.zip',
    'Ant-v3': 'https://huggingface.co/sb3/td3-Ant-v3/resolve/main/td3-Ant-v3.zip'
}

print("Downloading expert models from Hugging Face...")
print("=" * 60)

for env_name, url in models.items():
    output_file = f'experts/td3-{env_name}.zip'
    print(f"\nDownloading {env_name}...")
    print(f"URL: {url}")
    print(f"Saving to: {output_file}")

    try:
        urllib.request.urlretrieve(url, output_file)
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
        print(f"[OK] Downloaded successfully! Size: {file_size:.2f} MB")
    except Exception as e:
        print(f"[ERROR] Error downloading {env_name}: {e}")

print("\n" + "=" * 60)
print("Download complete!")

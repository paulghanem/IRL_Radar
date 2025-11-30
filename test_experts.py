"""
Test script to verify expert models are properly loaded for all MuJoCo environments.
"""

import os
import sys
from utils.helpers import GenerateDemo

def test_expert_loading(env_name, max_frames=100):
    """Test if expert can be loaded and generate demonstrations."""
    print(f"\n{'='*60}")
    print(f"Testing: {env_name}")
    print('='*60)

    try:
        demo_generator = GenerateDemo(env_name, max_frames=max_frames)
        states, actions, rewards, env = demo_generator.generate_demo(seed=123)

        print(f"[PASS] Successfully loaded expert for {env_name}")
        print(f"  - State shape: {states.shape}")
        print(f"  - Action shape: {actions.shape}")
        print(f"  - Final reward: {rewards[-1]:.2f}")
        print(f"  - Number of steps: {len(states)}")

        return True
    except Exception as e:
        print(f"[FAIL] Failed to load expert for {env_name}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # List of MuJoCo environments to test
    envs_to_test = [
        "Walker2d",
        "Hopper",
        "HalfCheetah-v4",
        "Swimmer"
    ]

    results = {}
    for env_name in envs_to_test:
        results[env_name] = test_expert_loading(env_name, max_frames=100)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for env_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {env_name}")

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

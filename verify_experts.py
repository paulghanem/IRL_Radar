"""
Quick verification script to check if all experts can be loaded.
"""

import os
from utils.helpers import GenerateDemo

def quick_check(env_name):
    """Quick check if expert can be loaded."""
    try:
        demo_generator = GenerateDemo(env_name, max_frames=10)  # Only 10 frames
        states, actions, rewards, env = demo_generator.generate_demo(seed=123)
        print(f"[PASS] {env_name:20s} - State dim: {states.shape[1]}, Action dim: {actions.shape[1]}, Final reward: {rewards[-1]:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] {env_name:20s} - Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPERT VERIFICATION TEST")
    print("="*70)

    envs = ["Walker2d", "Hopper", "HalfCheetah-v4", "Swimmer"]
    results = {env: quick_check(env) for env in envs}

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if all(results.values()):
        print("\n[SUCCESS] All experts are properly integrated!")
    else:
        print("\n[WARNING] Some experts failed to load.")

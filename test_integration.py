"""
Test script to verify expert model integration with main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import GenerateDemo
import numpy as np

# Test all 5 working expert environments
test_environments = [
    'HalfCheetah-v4',
    'Walker2d-v4',
    'Hopper-v4',
    'Swimmer-v4',
    'Walker2d',  # Test alias
]

print("="*70)
print("Testing Expert Model Integration")
print("="*70)

results = []

for env_name in test_environments:
    print(f"\n{'='*70}")
    print(f"Testing: {env_name}")
    print(f"{'='*70}")

    try:
        # Create demo generator
        demo_generator = GenerateDemo(env_name, max_frames=100)

        # Generate demo
        states, actions, rewards, env = demo_generator.generate_demo(seed=123)

        # Check results
        print(f"\n[SUCCESS] {env_name}")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Final cumulative reward: {rewards[-1]:.2f}")
        print(f"  Number of steps: {len(states)}")

        results.append({
            'env': env_name,
            'success': True,
            'states_shape': states.shape,
            'actions_shape': actions.shape,
            'reward': rewards[-1],
            'steps': len(states)
        })

    except Exception as e:
        print(f"\n[FAILED] {env_name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

        results.append({
            'env': env_name,
            'success': False,
            'error': str(e)
        })

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"\nSuccessful: {len(successful)}/{len(results)}")
for r in successful:
    print(f"  ✓ {r['env']}: {r['steps']} steps, reward={r['reward']:.2f}")

if failed:
    print(f"\nFailed: {len(failed)}/{len(results)}")
    for r in failed:
        print(f"  ✗ {r['env']}: {r.get('error', 'Unknown error')}")

print("\n" + "="*70)
if len(successful) == len(results):
    print("ALL TESTS PASSED! Integration successful.")
else:
    print(f"SOME TESTS FAILED: {len(failed)} failures")
print("="*70)

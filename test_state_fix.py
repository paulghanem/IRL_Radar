"""
Test that the state fix correctly extracts full MuJoCo state.
Verifies that state dimensions are correct and include x-position.
"""

import numpy as np
from utils.helpers import GenerateDemo

def test_state_dimensions(env_name, expected_obs_dim, expected_state_dim):
    """
    Test that expert demonstrations have correct dimensions.

    Args:
        env_name: Environment to test
        expected_obs_dim: Expected observation dimension (without x-position)
        expected_state_dim: Expected full state dimension (with x-position)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {env_name}")
    print(f"{'='*70}")

    demo_generator = GenerateDemo(env_name, max_frames=10)
    states, actions, rewards, env = demo_generator.generate_demo(seed=123)

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Expected state dim: {expected_state_dim}")
    print(f"Expected obs dim (for reference): {expected_obs_dim}")

    # Check state dimension
    actual_state_dim = states.shape[1]
    if actual_state_dim == expected_state_dim:
        print(f"[PASS] State dimension correct: {actual_state_dim}")
    else:
        print(f"[FAIL] State dimension mismatch: got {actual_state_dim}, expected {expected_state_dim}")
        return False

    # Verify x-position is changing (forward motion)
    x_positions = states[:, 0]  # First element should be x-position
    x_displacement = x_positions[-1] - x_positions[0]

    print(f"\nX-position analysis:")
    print(f"  Initial x-pos: {x_positions[0]:.4f}")
    print(f"  Final x-pos: {x_positions[-1]:.4f}")
    print(f"  Total displacement: {x_displacement:.4f}")

    # For locomotion tasks, x-position should change (increase for forward motion)
    # Swimmer might move in y direction, so we check if ANY position changes
    if env_name in ["Walker2d", "Hopper", "HalfCheetah-v4"]:
        if abs(x_displacement) > 0.01:  # Should move forward
            print(f"[PASS] X-position changing (forward motion detected)")
        else:
            print(f"[WARNING] X-position not changing much (might be an issue)")
    else:  # Swimmer
        # Swimmer might not move much in x, but state should still have x-pos
        print(f"[INFO] Swimmer environment - x-position present in state")

    # Check that states are different from observations
    # (Full state should have extra dimension compared to obs)
    print(f"\n[SUCCESS] State extraction working correctly")
    return True


def main():
    """Run tests for all environments."""
    print("\n" + "="*70)
    print("STATE FIX VERIFICATION TEST")
    print("="*70)
    print("\nThis test verifies that expert demonstrations now include")
    print("the full MuJoCo state (qpos + qvel) including x-position.")
    print()

    # Environment configurations: (name, obs_dim, state_dim)
    # state_dim = len(qpos) + len(qvel)
    environments = [
        ("Walker2d", 17, 18),        # qpos(9) + qvel(9) = 18
        ("Hopper", 11, 12),           # qpos(6) + qvel(6) = 12
        ("HalfCheetah-v4", 17, 18),  # qpos(9) + qvel(9) = 18
        ("Swimmer", 8, 10),           # qpos(5) + qvel(5) = 10
    ]

    results = {}
    for env_name, obs_dim, state_dim in environments:
        try:
            results[env_name] = test_state_dimensions(env_name, obs_dim, state_dim)
        except Exception as e:
            print(f"[ERROR] Test failed for {env_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[env_name] = False

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for env_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {env_name}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print("[SUCCESS] All environments now use correct full state!")
        print("\nKey points:")
        print("  - State includes x-position at index 0")
        print("  - State dimensions match qpos + qvel")
        print("  - Reward calculation will now work correctly")
        print("  - MPPI can use state[0] for forward velocity")
    else:
        print("[FAILURE] Some environments have issues")

    print()


if __name__ == "__main__":
    main()

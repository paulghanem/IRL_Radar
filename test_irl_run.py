"""
Quick test to verify IRL training runs correctly with integrated experts.
This script runs a minimal IRL training loop to ensure everything works end-to-end.
"""

import subprocess
import sys

def test_irl_run(env_name="Walker2d", method="rgcl", n_steps=50, iterations=2):
    """
    Test IRL training with minimal settings.

    Args:
        env_name: Environment to test (Walker2d, Hopper, HalfCheetah-v4, Swimmer)
        method: IRL method (rgcl, gail, airl)
        n_steps: Number of steps per trajectory
        iterations: Number of IRL iterations
    """
    print(f"\n{'='*70}")
    print(f"Testing IRL Training: {env_name} with {method.upper()}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "main.py",
        "--gym_env", env_name,
        f"--{method}",
        "--N_steps", str(n_steps),
        "--N_steps_expert", str(n_steps),
        "--rirl_iterations", str(iterations),
        "--experiment_name", f"test_{env_name}_{method}",
        "--seed", "123",
        "--reward_fn_updates", "5",  # Fewer updates for faster test
        "--horizon", "20",  # Smaller horizon for faster MPPI
        "--num_traj", "500",  # Fewer trajectories for faster MPPI
    ]

    print(f"Running command:")
    print(f"  {' '.join(cmd)}\n")

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"\n{'='*70}")
            print(f"[SUCCESS] IRL training completed successfully!")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"[FAIL] IRL training failed with return code {result.returncode}")
            print(f"{'='*70}\n")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n{'='*70}")
        print(f"[TIMEOUT] IRL training exceeded 5 minute timeout")
        print(f"{'='*70}\n")
        return False
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"[ERROR] IRL training failed with exception: {str(e)}")
        print(f"{'='*70}\n")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IRL INTEGRATION TEST")
    print("="*70)
    print("\nThis test will run a minimal IRL training loop to verify that:")
    print("  1. Expert demonstrations load correctly")
    print("  2. Cost function learns from demonstrations")
    print("  3. Policy generates trajectories")
    print("  4. IRL update loop works end-to-end")
    print("\nSettings:")
    print("  - Environment: Walker2d")
    print("  - Algorithm: RGCL")
    print("  - Steps: 50 per trajectory")
    print("  - Iterations: 2")
    print("  - Expected duration: ~2-3 minutes")
    print()

    # Run the test
    success = test_irl_run(
        env_name="Walker2d",
        method="rgcl",
        n_steps=50,
        iterations=2
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)

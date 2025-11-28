"""
Test PPO integration with main.py IRL loop
"""

import subprocess
import sys

print("="*60)
print("Testing PPO integration with IRL main loop")
print("="*60)

# Test with PPO on CartPole for a few iterations
cmd = [
    "C:/Users/siliconsynapse/anaconda3/envs/rirl/python.exe",
    "main.py",
    "--gym_env", "CartPole-v1",
    "--PPO",  # BooleanOptionalAction doesn't take a value
    "--rirl_iterations", "100",
    "--N_steps", "1000",
    "--reward_fn_updates", "10",
    "--hidden_dim", "64",
    "--seed", "42"
]

print("\nRunning command:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(
        cmd,
        cwd="C:/Users/siliconsynapse/Desktop/IRL_Radar",
        capture_output=True,
        text=True,
        timeout=18000
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode == 0:
        print("\n" + "="*60)
        print("SUCCESS: PPO integration test passed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"FAILED: Process exited with code {result.returncode}")
        print("="*60)
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("\n" + "="*60)
    print("TIMEOUT: Test took too long (>180s)")
    print("="*60)
    sys.exit(1)

except Exception as e:
    print("\n" + "="*60)
    print(f"ERROR: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
    sys.exit(1)

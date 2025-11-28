"""
Test script for RGCL with PPO implementation
Quick test to verify the RGCL+PPO combination works correctly
"""

import subprocess
import sys

print("="*60)
print("Testing RGCL with PPO Implementation")
print("="*60)

# Run a quick test with minimal iterations
cmd = [
    sys.executable, "main_ppo_rgcl.py",
    "--gym_env", "CartPole-v1",
    "--rirl_iterations", "5",
    "--N_steps", "50",
    "--N_steps_expert", "200",
    "--hidden_dim", "32",
    "--ppo_lr", "3e-4",
    "--P", "1e-2",
    "--Q", "1e-4",
    "--seed", "42"
]

print("\nRunning command:")
print(" ".join(cmd))
print("\n" + "="*60 + "\n")

result = subprocess.run(cmd)

print("\n" + "="*60)
if result.returncode == 0:
    print("✓ Test completed successfully!")
else:
    print("✗ Test failed with return code:", result.returncode)
print("="*60)

sys.exit(result.returncode)

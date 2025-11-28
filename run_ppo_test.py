"""
Quick test run of IRL with PPO on CartPole (few iterations for testing)
"""

import subprocess
import sys

# Run main_ppo.py with a small number of iterations for testing
cmd = [
    sys.executable,
    "main_ppo.py",
    "--gym_env", "CartPole-v1",
    "--rirl_iterations", "20",  # Just 20 iterations for quick test
    "--N_steps", "200",
    "--N_steps_expert", "200",
    "--reward_fn_updates", "5",  # Fewer updates per iteration
    "--lr", "1e-3",
    "--ppo_lr", "3e-4",
    "--rollout_length", "200",
    "--hidden_dim", "64",
    "--seed", "42",
    "--experiment_name", "test_ppo_cartpole"
]

print("="*60)
print("Running PPO-IRL test on CartPole...")
print("Command:", " ".join(cmd))
print("="*60)
print()

result = subprocess.run(cmd)
sys.exit(result.returncode)

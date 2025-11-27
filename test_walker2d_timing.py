"""
Simple timing test for Walker2d with specified parameters
"""
import subprocess
import time
import sys

# Time the execution
start_time = time.time()

# Run main.py with specified parameters
cmd = [
    "python", "main.py",
    "--gym_env=Walker2d",
    "--num_traj=500",
    "--horizon=50",
    "--N_steps=100",
    "--N_steps_expert=100",
    "--rirl_iterations=1",
    "--reward_fn_updates=1",
    "--UB",
    "--seed=123",
    "--lr=1e-4",
    "--lambda_=0.01",
    "--Q=1e-4",
    "--P=1e-2",
    "--hidden_dim=16",
    "--no-save_images"
]

print("="*80)
print("Running Walker2d with parameters:")
print("  - Environment: Walker2d")
print("  - MPPI trajectories (num_traj): 500")
print("  - Horizon: 50")
print("  - Time steps (N_steps): 100")
print("  - Iterations: 1 (single episode)")
print("="*80)
print()

result = subprocess.run(cmd, capture_output=False)

end_time = time.time()
elapsed_time = end_time - start_time

print()
print("="*80)
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("="*80)

sys.exit(result.returncode)

#!/usr/bin/env python3
"""
Benchmark loop vs lax version - Small test first, then full benchmark on GPU
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
from argparse import ArgumentParser

# Parse command line args
parser = ArgumentParser()
parser.add_argument('--test', action='store_true', help='Run small test first (N_steps=5, horizon=5, samples=5 on CPU)')
parser.add_argument('--platform', default='cuda', choices=['cpu', 'cuda'], help='Platform: cpu or cuda')
args_cmd = parser.parse_args()

# Set platform
os.environ['JAX_PLATFORMS'] = args_cmd.platform
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

print("=" * 80)
print(f"Benchmark: generate_session_loop vs generate_session_lax")
print("=" * 80)
print(f"Platform: {args_cmd.platform.upper()}")
print(f"JAX devices: {jax.devices()}")
print()

# Set parameters based on test vs full
if args_cmd.test:
    N_STEPS = 5
    HORIZON = 5
    NUM_SAMPLES = 5
    print("MODE: Small Test")
else:
    N_STEPS = 100
    HORIZON = 50
    NUM_SAMPLES = 500
    print("MODE: Full Benchmark")

print(f"Parameters:")
print(f"  N_steps: {N_STEPS}")
print(f"  horizon: {HORIZON}")
print(f"  num_samples: {NUM_SAMPLES}")
print(f"  Environment: Walker2d")
print("=" * 80)
print()

# Now run main.py with these settings
sys.argv = [
    'main.py',
    f'--horizon={HORIZON}',
    f'--N_steps={N_STEPS}',
    f'--gym_env=Walker2d',
    '--lr=1e-4',
    f'--num_traj={NUM_SAMPLES}',
    '--reward_fn_updates=1',
    '--lambda_=0.01',
    '--rirl_iterations=1',
    '--UB',
    '--no-save_images',
    '--seed=123'
]

# Patch the generate_session calls in main.py to benchmark both
original_code = open('main.py', 'r').read()

# Find where generate_session is called and add timing
benchmark_code = original_code.replace(
    "trajs=[policy.generate_session_loop(args,state_train,D_demo)]",
    """
import time
print("\\n" + "="*80)
print("BENCHMARKING LOOP VERSION")
print("="*80)
jax.block_until_ready(state_train.params)  # Ensure compilation is done
start_loop = time.time()
trajs_loop = [policy.generate_session_loop(args, state_train, D_demo)]
jax.block_until_ready(trajs_loop[0][0])  # Block until complete
end_loop = time.time()
loop_time = end_loop - start_loop
print(f"✓ Loop version completed in {loop_time:.4f} seconds")
print(f"  Return: {trajs_loop[0][3]}")

print("\\n" + "="*80)
print("BENCHMARKING LAX VERSION")
print("="*80)
start_lax = time.time()
trajs_lax = [policy.generate_session_lax(args, state_train, D_demo)]
jax.block_until_ready(trajs_lax[0][0])  # Block until complete
end_lax = time.time()
lax_time = end_lax - start_lax
print(f"✓ LAX version completed in {lax_time:.4f} seconds")
print(f"  Return: {trajs_lax[0][3]}")

print("\\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)
print(f"Loop version: {loop_time:.4f} seconds")
print(f"LAX version:  {lax_time:.4f} seconds")
speedup = loop_time / lax_time
print(f"Speedup: {speedup:.2f}x {'(LAX faster)' if speedup > 1 else '(Loop faster)'}")
print("="*80)

# Use loop version for rest of code
trajs = trajs_loop
"""
)

# Execute the modified code
exec(benchmark_code, globals())

#!/usr/bin/env python3
"""
Benchmark loop vs lax version of generate_session
Tests with the user's requested parameters:
- N_steps = 100
- horizon = 50
- num_samples = 500
"""
import sys
import time
sys.argv = ['main.py', '--horizon=50', '--N_steps=100', '--gym_env=Walker2d', '--lr=1e-4', '--num_traj=500',
            '--reward_fn_updates=1', '--lambda_=0.01', '--rirl_iterations=1', '--UB', '--no-save_images', '--seed=123']

# Import main to get all the setup
import main as main_module
import jax
print("="*80)
print("BENCHMARK: generate_session_loop vs generate_session_lax")
print("="*80)
print(f"JAX devices: {jax.devices()}")
print()
print("Parameters:")
print(f"  N_steps: 100")
print(f"  horizon: 50")
print(f"  num_samples: 500")
print(f"  Environment: Walker2d")
print(f"  Single trajectory (1 iteration)")
print("="*80)
print()

# Get the policy object from the running code
exec(open('main.py').read(), main_module.__dict__)

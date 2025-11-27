#!/usr/bin/env python3
"""
Benchmark: Loop version only
"""
import os
import sys
import time

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cuda'

# Set arguments for main.py
sys.argv = [
    'main.py',
    '--horizon=50',
    '--N_steps=100',
    '--gym_env=Walker2d',
    '--lr=1e-4',
    '--num_traj=500',
    '--reward_fn_updates=1',
    '--lambda_=0.01',
    '--rirl_iterations=1',
    '--UB',
    '--no-save_images',
    '--seed=123'
]

# Monkey-patch to add timing
import jax
original_main = None

def timed_wrapper():
    """Run main with timing around generate_session_loop"""
    import main as main_module

    # Store original function
    from src.control.mppi_class import MPPI
    original_generate = MPPI.generate_session_loop

    # Create wrapper that times the function
    def timed_generate_session_loop(self, args, state_train, D_demo):
        print("\n" + "="*80)
        print("TIMING: generate_session_loop (FOR LOOP VERSION)")
        print("="*80)
        jax.block_until_ready(state_train.params)
        start = time.time()
        result = original_generate(self, args, state_train, D_demo)
        jax.block_until_ready(result[0])
        end = time.time()
        elapsed = end - start
        print(f"✓ Loop version completed in {elapsed:.4f} seconds")
        print(f"  Return: {result[3]}")
        print("="*80 + "\n")
        return result

    # Patch the method
    MPPI.generate_session_loop = timed_generate_session_loop

    # Run main
    exec(open('main.py').read(), main_module.__dict__)

if __name__ == '__main__':
    timed_wrapper()

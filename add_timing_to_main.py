#!/usr/bin/env python3
"""
Add timing instrumentation to main.py to benchmark loop vs lax
"""
import sys

# Read main.py
with open('main.py', 'r') as f:
    content = f.read()

# Find the line with generate_session_loop and add timing around both versions
old_code = "                trajs=[policy.generate_session_loop(args,state_train,D_demo)]"

new_code = '''                import time
                import jax

                # Benchmark LOOP version
                print("\\n" + "="*80)
                print("TIMING: generate_session_loop")
                print("="*80)
                jax.block_until_ready(state_train.params)  # Warm up
                start_loop = time.time()
                trajs_loop = [policy.generate_session_loop(args, state_train, D_demo)]
                jax.block_until_ready(trajs_loop[0][0])
                end_loop = time.time()
                loop_time = end_loop - start_loop
                print(f"Loop version: {loop_time:.4f} seconds (Return: {trajs_loop[0][3]})")

                # Benchmark LAX version
                print("\\n" + "="*80)
                print("TIMING: generate_session_lax")
                print("="*80)
                start_lax = time.time()
                trajs_lax = [policy.generate_session_lax(args, state_train, D_demo)]
                jax.block_until_ready(trajs_lax[0][0])
                end_lax = time.time()
                lax_time = end_lax - start_lax
                print(f"LAX version:  {lax_time:.4f} seconds (Return: {trajs_lax[0][3]})")

                # Report
                print("\\n" + "="*80)
                print("BENCHMARK RESULT")
                print("="*80)
                print(f"Loop: {loop_time:.4f}s  |  LAX: {lax_time:.4f}s  |  Speedup: {loop_time/lax_time:.2f}x")
                print("="*80 + "\\n")

                # Use loop for rest of code
                trajs = trajs_loop'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('main_with_timing.py', 'w') as f:
        f.write(content)
    print("✓ Created main_with_timing.py with benchmark instrumentation")
else:
    print("✗ Could not find the target line in main.py")
    sys.exit(1)

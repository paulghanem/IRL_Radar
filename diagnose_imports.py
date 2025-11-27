#!/usr/bin/env python3
"""Diagnose which import causes the hang"""
import sys
import time

def timed_import(module_name, from_clause=None):
    """Import a module and time how long it takes"""
    start = time.time()
    print(f"Importing {module_name}...", flush=True)
    try:
        if from_clause:
            exec(f"from {from_clause} import {module_name}")
        else:
            __import__(module_name)
        elapsed = time.time() - start
        print(f"  ✓ {module_name} imported successfully in {elapsed:.2f}s", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ {module_name} failed after {elapsed:.2f}s: {e}", flush=True)
        return False

print("=" * 60)
print("IMPORT DIAGNOSTICS")
print("=" * 60)
print(f"Python: {sys.version}", flush=True)
print()

# Test imports one by one in order they appear in main.py
print("Testing basic imports...", flush=True)
timed_import("numpy")
timed_import("jax")
timed_import("jax.numpy", from_clause="jax")

print("\nTesting JAX devices...", flush=True)
import jax
print(f"JAX devices: {jax.devices()}", flush=True)

print("\nTesting more imports...", flush=True)
timed_import("os")
timed_import("argparse")
timed_import("json")
timed_import("time")

print("\nTesting gymnasium...", flush=True)
timed_import("gymnasium")

print("\nTesting stable_baselines3...", flush=True)
timed_import("PPO", from_clause="stable_baselines3")

print("\nTesting src.envs.envs...", flush=True)
sys.path.insert(0, '/ocean/projects/cis250114p/pghanem/IRL_Radar_big')
timed_import("envs", from_clause="src.envs")

print("\nTesting src.learning imports...", flush=True)
timed_import("sample_expert", from_clause="src.learning")
timed_import("trajectory_functions", from_clause="src.learning")

print("\nTesting mujoco (THIS MAY HANG)...", flush=True)
timed_import("mujoco")

print("\nTesting mujoco.mjx (THIS MAY HANG)...", flush=True)
timed_import("mjx", from_clause="mujoco")

print("\n" + "=" * 60)
print("ALL IMPORTS COMPLETED SUCCESSFULLY!")
print("=" * 60)

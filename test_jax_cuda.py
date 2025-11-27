#!/usr/bin/env python3
"""Simple test to verify JAX+CUDA is working"""
import time
print("Starting JAX+CUDA test...")
print(f"Time: {time.time()}")

print("Importing jax...")
import jax
print(f"JAX imported. Time: {time.time()}")

print("Importing jax.numpy...")
import jax.numpy as jnp
print(f"jax.numpy imported. Time: {time.time()}")

print("Checking devices...")
devices = jax.devices()
print(f"Devices: {devices}")
print(f"Device type: {devices[0].platform}")

print("\nTesting simple JAX operation...")
x = jnp.array([1, 2, 3, 4, 5])
y = x * 2
print(f"Result: {y}")

print("\nJAX+CUDA test completed successfully!")

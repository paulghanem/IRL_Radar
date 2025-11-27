#!/usr/bin/env python3
"""Test JAX float64 downcasting behavior when x64 is disabled"""

import jax
import jax.numpy as jnp

print("=" * 60)
print("Testing JAX float64 downcast behavior")
print("=" * 60)
print(f"jax.config.x64_enabled: {jax.config.x64_enabled}")
print()

# Explicitly request float64
arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print(f"Requested dtype: jnp.float64")
print(f"Actual dtype:    {arr.dtype}")
print()

# Try reshaping with float64
arr2 = jnp.array([[1.0, 2.0]].reshape((1,-1)), dtype=jnp.float64)
print(f"Array with reshape, dtype=jnp.float64:")
print(f"  Actual dtype: {arr2.dtype}")
print()

print("=" * 60)
print("CONCLUSION:")
if arr.dtype == jnp.float32:
    print("  float64 requests are DOWNCAST to float32 (x64 disabled)")
else:
    print(f"  Arrays are actually {arr.dtype}")
print("=" * 60)

#!/usr/bin/env python3
"""Quick script to check JAX dtype configuration"""

import jax
import jax.numpy as jnp

print("=" * 60)
print("JAX DTYPE CONFIGURATION CHECK")
print("=" * 60)
print()

# Check JAX configuration
print("JAX Configuration:")
print(f"  JAX version: {jax.__version__}")
print(f"  jax.config.x64_enabled: {jax.config.x64_enabled}")
print()

# Check default dtypes
print("Default dtypes:")
x = jnp.array([1.0, 2.0, 3.0])
print(f"  jnp.array([1.0, 2.0, 3.0]).dtype: {x.dtype}")

y = jnp.ones(5)
print(f"  jnp.ones(5).dtype: {y.dtype}")

z = jnp.random.normal(jax.random.PRNGKey(0), (3, 3))
print(f"  jnp.random.normal(...).dtype: {z.dtype}")

# Check computation dtype
a = jnp.array([1.5, 2.5])
b = jnp.array([3.5, 4.5])
result = a + b
print(f"  (array + array).dtype: {result.dtype}")

# Check what MJX uses
try:
    import mujoco
    from mujoco import mjx
    print()
    print("MuJoCo/MJX Configuration:")
    print(f"  MuJoCo version: {mujoco.__version__}")
    print("  MJX uses JAX default dtypes")
except ImportError:
    print()
    print("MuJoCo/MJX not available in quick check")

print()
print("=" * 60)
print("SUMMARY:")
if jax.config.x64_enabled:
    print("  Running in FP64 (float64) mode")
else:
    print("  Running in FP32 (float32) mode [DEFAULT]")
print("=" * 60)

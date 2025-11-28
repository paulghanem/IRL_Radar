"""Quick test to debug state dimensions"""
import jax.numpy as jnp
import numpy as np
from src.control.simplified_walker import SimplifiedWalker

env = SimplifiedWalker()

# Get initial state
state = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"Initial state: {state}")

# Test step
action = jnp.zeros(6)
next_state = env.step(state, action)
print(f"\nNext state shape: {next_state.shape}")
print(f"Next state: {next_state}")

# Detailed breakdown
print(f"\nBreakdown:")
print(f"  x, z: {next_state[0:2]}")
print(f"  angles (8): {next_state[2:10]}")
print(f"  velocities (8): {next_state[10:18]}")

if next_state.shape[0] != 18:
    print(f"\nERROR: State has {next_state.shape[0]} dims, expected 18!")
    print(f"Extra dims: {next_state.shape[0] - 18}")
else:
    print("\nSUCCESS: State has correct 18 dimensions")

"""Unit test to verify hand-coded neural network matches Flax CostNN exactly."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

# Import the hand-coded neural network
import sys
sys.path.append('src/control')
from mppi_class import neural_net_forward_pass

# Import the original CostNN
from cost_jax import CostNN

def test_neural_net_equivalence():
    """Test that hand-coded NN produces identical outputs to Flax CostNN."""

    # Setup
    state_dim = 18  # Walker2d state dimension
    hidden_dim = 16
    batch_size = 100

    print("="*60)
    print("UNIT TEST: Neural Network Equivalence")
    print("="*60)

    # Create the Flax model
    cost_nn = CostNN(state_dims=state_dim, hidden_dim=hidden_dim)
    init_rng = jax.random.key(42)
    dummy_input = jnp.ones((1, state_dim))
    variables = cost_nn.init(init_rng, dummy_input)
    params = variables['params']

    print(f"\nModel architecture:")
    print(f"  Input dim: {state_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: 1")
    print(f"  Layers: Dense({hidden_dim})->ReLU->Dense({hidden_dim})->ReLU->Dense(1)->clip(x^2,0,5)")

    # Create test inputs
    test_rng = jax.random.key(123)
    test_states = jax.random.normal(test_rng, (batch_size, state_dim))

    print(f"\nTest batch size: {batch_size}")

    # Get outputs from Flax model
    flax_outputs = cost_nn.apply({'params': params}, test_states)

    # Get outputs from hand-coded model
    handcoded_outputs = neural_net_forward_pass(test_states, params)

    # Compare
    max_diff = jnp.max(jnp.abs(flax_outputs - handcoded_outputs))
    mean_diff = jnp.mean(jnp.abs(flax_outputs - handcoded_outputs))

    print(f"\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Check if they match (allowing for floating point precision)
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\n[SUCCESS] Networks are identical (within {tolerance} tolerance)")
        print(f"   Flax output sample: {flax_outputs[:3].ravel()}")
        print(f"   Handcoded output:   {handcoded_outputs[:3].ravel()}")
        return True
    else:
        print(f"\n[FAILURE] Networks differ by {max_diff:.2e}")
        print(f"   Flax output sample: {flax_outputs[:3].ravel()}")
        print(f"   Handcoded output:   {handcoded_outputs[:3].ravel()}")
        return False

def benchmark_speed_comparison():
    """Benchmark speed of Flax vs hand-coded implementation."""

    state_dim = 18
    hidden_dim = 16
    batch_size = 1500  # 500 samples * 3 (2 horizon + 1 terminal)

    print("\n" + "="*60)
    print("SPEED BENCHMARK:")
    print("="*60)

    # Create models
    cost_nn = CostNN(state_dims=state_dim, hidden_dim=hidden_dim)
    init_rng = jax.random.key(42)
    dummy_input = jnp.ones((1, state_dim))
    variables = cost_nn.init(init_rng, dummy_input)
    params = variables['params']

    # Create test data
    test_rng = jax.random.key(123)
    test_states = jax.random.normal(test_rng, (batch_size, state_dim))

    # Warm up (JIT compilation)
    print(f"\nWarming up (JIT compilation for {batch_size} states)...")
    _ = cost_nn.apply({'params': params}, test_states)
    _ = neural_net_forward_pass(test_states, params)

    # Benchmark Flax
    print("\nBenchmarking Flax apply_fn...")
    import time
    n_iterations = 100

    start = time.time()
    for _ in range(n_iterations):
        out = cost_nn.apply({'params': params}, test_states)
        out.block_until_ready()
    flax_time = (time.time() - start) / n_iterations * 1000  # ms

    # Benchmark hand-coded
    print("Benchmarking hand-coded neural_net_forward_pass...")
    start = time.time()
    for _ in range(n_iterations):
        out = neural_net_forward_pass(test_states, params)
        out.block_until_ready()
    handcoded_time = (time.time() - start) / n_iterations * 1000  # ms

    print(f"\n" + "="*60)
    print("SPEED RESULTS (average over {n_iterations} iterations):")
    print("="*60)
    print(f"Flax apply_fn:           {flax_time:.4f} ms")
    print(f"Hand-coded forward pass: {handcoded_time:.4f} ms")
    print(f"Speedup:                 {flax_time/handcoded_time:.2f}x")

    if handcoded_time < flax_time:
        print(f"\n[SUCCESS] Hand-coded is {flax_time/handcoded_time:.2f}x FASTER!")
    else:
        print(f"\n[WARNING] Flax is {handcoded_time/flax_time:.2f}x faster (unexpected)")

if __name__ == "__main__":
    # Run equivalence test
    success = test_neural_net_equivalence()

    if success:
        # Run speed benchmark
        benchmark_speed_comparison()
    else:
        print("\n[SKIPPED] Skipping benchmark due to equivalence test failure")

    print("\n" + "="*60)

"""Debug why hand-coded NN isn't fast in MPPI."""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import time
import sys
sys.path.append('src/control')
from mppi_class import neural_net_forward_pass, compute_all_costs_handcoded_nn
from cost_jax import CostNN

# Setup matching MPPI scenario
state_dim = 18
hidden_dim = 16
num_samples = 200  # Current setting
horizon = 2

print("="*60)
print("DEBUGGING COST COMPUTATION TIMING")
print("="*60)
print(f"\nMPPI Settings:")
print(f"  num_samples: {num_samples}")
print(f"  horizon: {horizon}")
print(f"  Total states to process: {num_samples * (horizon + 1)} = {num_samples * (horizon + 1)}")

# Create model and params
cost_nn = CostNN(state_dims=state_dim, hidden_dim=hidden_dim)
init_rng = jax.random.key(42)
dummy_input = jnp.ones((1, state_dim))
variables = cost_nn.init(init_rng, dummy_input)
params = variables['params']

# Create test data matching MPPI structure
test_rng = jax.random.key(123)
state_seq_batch = jax.random.normal(test_rng, (num_samples, horizon + 1, state_dim))

print(f"\nState batch shape: {state_seq_batch.shape}")

# Test 1: Original compute_all_costs_batched approach
print("\n" + "="*60)
print("Test 1: Flax apply_fn approach (original slow method)")
print("="*60)

def compute_costs_flax(state_seq_batch, params, apply_fn):
    num_samples, horizon_plus_1, state_dim = state_seq_batch.shape
    horizon = horizon_plus_1 - 1

    # Batch all trajectory states
    traj_states = state_seq_batch[:, :-1, :].reshape(-1, state_dim)
    traj_costs = apply_fn({'params': params}, traj_states).reshape(num_samples, horizon)

    # Terminal states
    terminal_costs = apply_fn({'params': params}, state_seq_batch[:, -1, :]).ravel()

    return jnp.sum(traj_costs, axis=1) + terminal_costs

# Warm up
_ = compute_costs_flax(state_seq_batch, params, cost_nn.apply)

# Time it
n_iters = 20
start = time.time()
for _ in range(n_iters):
    out = compute_costs_flax(state_seq_batch, params, cost_nn.apply)
    out.block_until_ready()
flax_time = (time.time() - start) / n_iters * 1000

print(f"Flax apply_fn: {flax_time:.4f} ms per iteration")

# Test 2: Hand-coded NN approach
print("\n" + "="*60)
print("Test 2: Hand-coded NN approach")
print("="*60)

# Warm up
_ = compute_all_costs_handcoded_nn(state_seq_batch, params)

# Time it
start = time.time()
for _ in range(n_iters):
    out = compute_all_costs_handcoded_nn(state_seq_batch, params)
    out.block_until_ready()
handcoded_time = (time.time() - start) / n_iters * 1000

print(f"Hand-coded NN: {handcoded_time:.4f} ms per iteration")

# Test 3: Just the neural network forward pass
print("\n" + "="*60)
print("Test 3: Just neural_net_forward_pass (no reshaping)")
print("="*60)

# Flatten all states
all_states = state_seq_batch.reshape(-1, state_dim)
print(f"All states shape: {all_states.shape}")

# Warm up
_ = neural_net_forward_pass(all_states, params)

# Time it
start = time.time()
for _ in range(n_iters):
    out = neural_net_forward_pass(all_states, params)
    out.block_until_ready()
pure_nn_time = (time.time() - start) / n_iters * 1000

print(f"Pure NN forward: {pure_nn_time:.4f} ms per iteration")

# Results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Flax apply_fn:          {flax_time:.4f} ms")
print(f"Hand-coded full:        {handcoded_time:.4f} ms  (speedup: {flax_time/handcoded_time:.2f}x)")
print(f"Hand-coded NN only:     {pure_nn_time:.4f} ms  (speedup: {flax_time/pure_nn_time:.2f}x)")

if handcoded_time >= flax_time * 0.9:
    print("\n[WARNING] Hand-coded is NOT faster! Investigating...")
    print("Possible issues:")
    print("  1. JIT compilation not working properly")
    print("  2. Params dict causing re-tracing")
    print("  3. Batch size too small to see benefits")
else:
    print(f"\n[SUCCESS] Hand-coded is {flax_time/handcoded_time:.2f}x faster!")

print("\n" + "="*60)

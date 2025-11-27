#!/bin/bash
#SBATCH --job-name=simple_walker_gpu
#SBATCH --output=simple_walker_direct_%j.out
#SBATCH --error=simple_walker_direct_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=00:30:00
#SBATCH --account=cis250114p

echo "=========================================="
echo "Direct GPU Test - Simplified Walker2d"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Show GPU info
nvidia-smi

# Load modules
module load anaconda3
source activate rirl

# Set environment for GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"

# Navigate to project directory
cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

echo ""
echo "Testing JAX GPU availability..."
python -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices()); print('Backend:', jax.default_backend())"

echo ""
echo "=========================================="
echo "Testing Simple Walker2d Dynamics"
echo "=========================================="

# Direct Python test
python << 'EOF'
import jax
import jax.numpy as jnp
import time

print(f"\nJAX Backend: {jax.default_backend()}")
print(f"JAX Devices: {jax.devices()}\n")

# Import simple Walker2d
from src.control.simple_walker2d import (
    simple_walker2d_step,
    simple_walker2d_reward,
    simple_walker2d_reset
)

# Test 1: Single trajectory
print("Test 1: Single 1000-step trajectory")
state = simple_walker2d_reset()
action = jnp.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])

# Warmup
for _ in range(10):
    state = simple_walker2d_step(state, action)
jax.block_until_ready(state)

# Actual test
state = simple_walker2d_reset()
start = time.time()
total_reward = 0.0
for i in range(1000):
    next_state = simple_walker2d_step(state, action)
    reward = simple_walker2d_reward(state, action, next_state)
    total_reward += float(reward)
    state = next_state
jax.block_until_ready(state)
elapsed = time.time() - start

print(f"  Time: {elapsed:.4f}s ({elapsed/1000*1000:.2f}ms per step)")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Final x position: {float(state[0]):.2f}\n")

# Test 2: Vectorized (parallel) execution
print("Test 2: Vectorized execution (500 parallel trajectories, 100 steps each)")
batch_size = 500
states = jnp.stack([simple_walker2d_reset() for _ in range(batch_size)])
actions = jnp.stack([action for _ in range(batch_size)])

# Warmup
next_states = jax.vmap(simple_walker2d_step)(states, actions)
jax.block_until_ready(next_states)

# Actual test
start = time.time()
for _ in range(100):
    next_states = jax.vmap(simple_walker2d_step)(states, actions)
    jax.block_until_ready(next_states)
    states = next_states
elapsed = time.time() - start
total_steps = batch_size * 100

print(f"  Time: {elapsed:.4f}s")
print(f"  Total steps: {total_steps}")
print(f"  Steps per second: {total_steps/elapsed:.0f}")
print(f"  Time per step: {elapsed/total_steps*1000:.4f}ms\n")

print("=" * 50)
print("GPU TEST SUCCESSFUL!")
print("=" * 50)
EOF

echo ""
echo "End time: $(date)"
echo "=========================================="

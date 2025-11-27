#!/bin/bash
# Test script to verify optimizations
# Compares performance before and after optimizations

echo "=========================================="
echo "Testing GPU Optimizations"
echo "=========================================="
echo ""

# Load modules
module load anaconda3/2024.10-1
conda activate rirl

# Set GPU environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_DISABLE_X64=1

# Show GPU info
echo "GPU Device:"
python -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
echo ""

# Test command
echo "Running PHASE 2 OPTIMIZED Walker2d test (1 iteration)..."
echo "Phase 1 optimizations: Removed CPU-GPU transfers, unused computations"
echo "Phase 2 optimizations: lax.scan rollouts + JIT reward updates"
echo ""
echo "Expected time: ~3-4 minutes (was 7.5 min)"
echo "Expected speedup: 50-60% faster"
echo ""

time python main.py \
    --seed=123 \
    --gym_env="Walker2d" \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=100 \
    --N_steps_expert=100 \
    --rirl_iterations=1 \
    --reward_fn_updates=15 \
    --lambda_=0.01 \
    --lr=1e-4 \
    --Q=1e-4 \
    --P=1e-2 \
    --hidden_dim=16 \
    --UB

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

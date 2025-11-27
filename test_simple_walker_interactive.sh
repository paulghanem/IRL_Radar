#!/bin/bash
# Interactive test script for simplified Walker2d
# Usage:
#   For GPU: salloc --gres=gpu:1 --partition=GPU-shared --account=cis250114p --time=01:00:00
#   Then run: bash test_simple_walker_interactive.sh gpu
#   Or for CPU: bash test_simple_walker_interactive.sh cpu

PLATFORM=${1:-cpu}

module load anaconda3
source activate rirl

if [ "$PLATFORM" = "gpu" ]; then
    echo "Running on GPU..."
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"
    nvidia-smi
else
    echo "Running on CPU..."
    export JAX_PLATFORMS=cpu
    export CUDA_VISIBLE_DEVICES=''
fi

echo "=========================================="
echo "Quick test with reduced parameters"
echo "=========================================="

python test_simple_walker2d.py \
    --platform $PLATFORM \
    --method gcl \
    --gym_env SimpleWalker2d \
    --seed 123 \
    --horizon 20 \
    --num_traj 100 \
    --N_steps 100 \
    --rirl_iterations 10 \
    --reward_fn_updates 5 \
    --lr 1e-4 \
    --lambda_ 0.01 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 16

echo "=========================================="
echo "Test complete!"
echo "=========================================="

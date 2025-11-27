#!/bin/bash
# Helper script to run code on GPU with proper environment setup

echo "=========================================="
echo "GPU Environment Setup"
echo "=========================================="

# Load modules
module load anaconda3/2024.10-1
source activate rirl

# Set required environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_DISABLE_X64=1

echo "Environment configured:"
echo "  LD_LIBRARY_PATH set"
echo "  JAX_DISABLE_X64=1"
echo ""

# Verify GPU is available
echo "Checking GPU availability..."
python -c "
import jax
print('✓ JAX version:', jax.__version__)
print('✓ JAX devices:', jax.devices())
print('✓ Backend:', jax.default_backend())

if jax.default_backend() == 'gpu':
    print('✓ GPU is ready!')
else:
    print('⚠️  Warning: Not running on GPU')
"

echo ""
echo "=========================================="
echo "Ready to run on GPU!"
echo "=========================================="
echo ""

# Check if a script was provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: ./run_on_gpu.sh <your_script.py> [args...]"
    echo ""
    echo "Examples:"
    echo "  ./run_on_gpu.sh main.py --seed=123 --gym_env=HalfCheetah-v4"
    echo "  ./run_on_gpu.sh benchmark_brax_mjx_fixed.py"
    echo ""
else
    echo "Running: python $@"
    echo "=========================================="
    echo ""
    python "$@"
fi

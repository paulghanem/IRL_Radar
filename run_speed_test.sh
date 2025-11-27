#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:30:00
#SBATCH --output=speed_test_%j.out
#SBATCH --error=speed_test_%j.err

echo "=== JIT Optimization Speed Test ==="
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl
echo "✓ Environment activated"
echo ""

# Configure CUDA library paths for JAX
NVIDIA_SITE_PACKAGES=$(python -c "import sys, os; paths = [p for p in sys.path if 'site-packages' in p and os.path.exists(os.path.join(p, 'nvidia'))]; print(paths[0] if paths else '')")

if [ -n "$NVIDIA_SITE_PACKAGES" ]; then
    NVIDIA_LIB_PATHS=$(find "$NVIDIA_SITE_PACKAGES/nvidia" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "==========================================="
echo "Test 1: Quick test with 10 steps"
echo "==========================================="
python test_speed_optimization.py \
    --gym_env=Walker2d \
    --horizon=5 \
    --num_traj=500 \
    --N_steps=10

echo ""
echo "==========================================="
echo "Test 2: Realistic test with 100 steps"
echo "==========================================="
python test_speed_optimization.py \
    --gym_env=Walker2d \
    --horizon=5 \
    --num_traj=500 \
    --N_steps=100

echo ""
echo "==========================================="
echo "Speed Test Complete"
echo "==========================================="

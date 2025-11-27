#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:30:00
#SBATCH -J test_simple_cpu
#SBATCH -o test_simple_walker2d_cpu_%j.out
#SBATCH -e test_simple_walker2d_cpu_%j.err

echo "=========================================="
echo "QUICK TEST - Simplified Walker2d GCL CPU"
echo "1 iteration, 10 steps"
echo "Started at: $(date)"
echo "=========================================="

module load anaconda3
source activate rirl

# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64=0

python test_simple_walker2d.py \
    --platform=cpu \
    --method=gcl \
    --gym_env=SimpleWalker2d \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=10 \
    --rirl_iterations=1 \
    --s_dim=18 \
    --a_dim=6

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

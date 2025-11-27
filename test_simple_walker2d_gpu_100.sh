#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:30:00
#SBATCH -J test_simple_gpu
#SBATCH -o test_simple_walker2d_gpu_100_%j.out
#SBATCH -e test_simple_walker2d_gpu_100_%j.err

echo "=========================================="
echo "GPU TEST - Simplified Walker2d GCL"
echo "1 iteration, 100 steps"
echo "Started at: $(date)"
echo "=========================================="

module load anaconda3
source activate rirl

# Use GPU (CUDA)
export JAX_PLATFORMS="cuda"

python test_simple_walker2d.py \
    --platform=cuda \
    --method=gcl \
    --gym_env=SimpleWalker2d \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=100 \
    --rirl_iterations=1 \
    --s_dim=18 \
    --a_dim=6

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

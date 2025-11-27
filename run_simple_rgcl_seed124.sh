#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 8:00:00
#SBATCH -J simple_rgcl_124
#SBATCH -o simple_rgcl_seed124_%j.out
#SBATCH -e simple_rgcl_seed124_%j.err

echo "=========================================="
echo "Simple Walker2d RGCL - Seed 124"
echo "1000 iterations, 1000 steps, sigma=1e-3"
echo "Started at: $(date)"
echo "=========================================="

module load anaconda3
source activate rirl

export JAX_PLATFORMS="cuda"

python test_simple_walker2d.py \
    --platform=cuda \
    --method=rgcl \
    --gym_env=SimpleWalker2d \
    --seed=124 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=1000 \
    --rirl_iterations=1000 \
    --Q=1e-5 \
    --P=1e-2 \
    --sigma=1e-3 \
    --s_dim=18 \
    --a_dim=6

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

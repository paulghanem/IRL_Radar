#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 8:00:00
#SBATCH -J rgcl_h20_126
#SBATCH -o simple_rgcl_h20_seed126_%j.out
#SBATCH -e simple_rgcl_h20_seed126_%j.err

echo "=========================================="
echo "Simple Walker2d RGCL - Horizon 20 - Seed 126"
echo "horizon=20, num_traj=1000, 1000 iterations"
echo "Started at: $(date)"
echo "=========================================="

module load anaconda3
source activate rirl

export JAX_PLATFORMS="cuda"

python test_simple_walker2d.py \
    --platform=cuda \
    --method=rgcl \
    --gym_env=SimpleWalker2d \
    --seed=126 \
    --horizon=20 \
    --num_traj=1000 \
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

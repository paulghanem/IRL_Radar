#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 0:30:00
#SBATCH -J test_learning
#SBATCH -o test_learning_%j.out
#SBATCH -e test_learning_%j.err

echo "=========================================="
echo "Quick Test: GCL with Learning (10 iterations)"
echo "Started at: $(date)"
echo "=========================================="

module load anaconda3
source activate rirl

export JAX_PLATFORMS="cuda"

python test_simple_walker2d.py \
    --platform=cuda \
    --method=gcl \
    --gym_env=SimpleWalker2d \
    --seed=123 \
    --horizon=20 \
    --num_traj=500 \
    --N_steps=100 \
    --rirl_iterations=10 \
    --reward_fn_updates=5 \
    --Q=1e-3 \
    --P=1e-2 \
    --sigma=1e-3 \
    --s_dim=18 \
    --a_dim=6

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

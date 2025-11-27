#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:30:00
#SBATCH -J test_simple_rgcl
#SBATCH -o test_simple_walker2d_rgcl_%j.out
#SBATCH -e test_simple_walker2d_rgcl_%j.err

# Quick test: 1 iteration, 100 steps to check execution time

cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

module load anaconda3
source activate /jet/home/pghanem/.conda/envs/rirl

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=========================================="
echo "QUICK TEST - Simplified Walker2d RGCL"
echo "1 iteration, 100 steps"
echo "Started at: $(date)"
echo "=========================================="

python test_simple_walker2d.py \
    --method=rgcl \
    --gym_env=SimpleWalker2d \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=100 \
    --rirl_iterations=1 \
    --reward_fn_updates=15 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=1e-5 \
    --P=1e-2 \
    --hidden_dim=16 \
    --s_dim=18 \
    --a_dim=6 \
    --dt=0.002 \
    --frame_skip=5

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

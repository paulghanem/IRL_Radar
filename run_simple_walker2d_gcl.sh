#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 48:00:00
#SBATCH -J simple_w2d_gcl
#SBATCH -o simple_walker2d_gcl_%j.out
#SBATCH -e simple_walker2d_gcl_%j.err

# Test Simplified Walker2d with GCL
# Parameters: horizon=50, num_traj=500, N_steps=1000, iterations=1000

cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

# Load modules
module load anaconda3
source activate /jet/home/pghanem/.conda/envs/rirl

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=========================================="
echo "Testing Simplified Walker2d with GCL"
echo "Started at: $(date)"
echo "=========================================="

python test_simple_walker2d.py \
    --method=gcl \
    --gym_env=SimpleWalker2d \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=1000 \
    --rirl_iterations=1000 \
    --reward_fn_updates=15 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --hidden_dim=16 \
    --s_dim=18 \
    --a_dim=6 \
    --dt=0.002 \
    --frame_skip=5

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

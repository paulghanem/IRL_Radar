#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 00:30:00
#SBATCH --output=walker2d_2iter_%j.out
#SBATCH --error=walker2d_2iter_%j.err

echo "========================================"
echo "Walker2d 2-Iteration Timing Test"
echo "========================================"
echo "Running on node: $(hostname -s)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Load modules
module load anaconda3/2024.10-1
source activate rirl

# Set environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_DISABLE_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export TF_CUDNN_USE_AUTOTUNE=1

echo "Starting experiment at: $(date)"
echo ""

# Run with 2 iterations
time python main.py \
    --gym_env=Walker2d \
    --num_traj=500 \
    --horizon=50 \
    --N_steps=100 \
    --rirl_iterations=2 \
    --reward_fn_updates=15 \
    --lambda_=0.01 \
    --seed=123 \
    --UB \
    --no-save_images

echo ""
echo "========================================"
echo "Experiment completed at: $(date)"
echo "========================================"

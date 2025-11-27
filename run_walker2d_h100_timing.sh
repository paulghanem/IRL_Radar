#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --gpus=h100-80:1
#SBATCH -t 00:20:00
#SBATCH --output=walker2d_h100_timing_%j.out
#SBATCH --error=walker2d_h100_timing_%j.err

# Load conda
module load anaconda3/2024.10-1
conda activate rirl

cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

echo "========================================"
echo "Walker2d H100 GPU MJX Timing Test"
echo "========================================"
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# GPU Information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Time the execution
echo "Starting experiment at: $(date)"
START_TIME=$(date +%s)

# Run the experiment - same parameters as before
python main.py \
    --gym_env=Walker2d \
    --num_traj=500 \
    --horizon=50 \
    --N_steps=100 \
    --N_steps_expert=100 \
    --rirl_iterations=1 \
    --reward_fn_updates=15 \
    --UB \
    --seed=123 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=1e-4 \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "Experiment completed at: $(date)"
echo "Total wall time: ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"
echo "========================================"
echo ""
echo "Verifying MJX usage (env_brax should be None for Walker2d):"
grep -A 5 "Walker2d" main.py | grep "env_brax"

#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 02:00:00
#SBATCH --output=regenerate_walker2d_%j.out
#SBATCH --error=regenerate_walker2d_%j.err

# Load conda
module load anaconda3/2024.10-1
conda activate rirl

echo "========================================"
echo "Regenerating Walker2d Expert Model"
echo "========================================"
echo "Running on node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

echo "Starting training at: $(date)"
python regenerate_walker2d_expert.py
echo "Training completed at: $(date)"
echo "========================================"

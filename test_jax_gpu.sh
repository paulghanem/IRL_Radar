#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 00:05:00
#SBATCH --output=jax_cuda_test_%j.out
#SBATCH --error=jax_cuda_test_%j.err

# Load conda
module load anaconda3/2024.10-1
conda activate rirl

echo "========================================"
echo "JAX+CUDA Compatibility Test"
echo "========================================"
echo "Running on node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Starting test at: $(date)"
python test_jax_cuda.py
echo "Test completed at: $(date)"
echo "========================================"

#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 00:10:00
#SBATCH --output=diagnostics_%j.out
#SBATCH --error=diagnostics_%j.err

# Load conda
module load anaconda3/2024.10-1
conda activate rirl

echo "========================================="
echo "Import Diagnostics"
echo "========================================="
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# Run with timeout to prevent indefinite hang
timeout 180 python diagnose_imports.py

exit_code=$?
echo ""
echo "Exit code: $exit_code"
if [ $exit_code -eq 124 ]; then
    echo "TIMEOUT: Script hung for more than 3 minutes"
fi
echo "End: $(date)"
echo "========================================="

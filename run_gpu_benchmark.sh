#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH --output=benchmark_gpu.%j.out
#SBATCH --error=benchmark_gpu.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ghanem.p@northeastern.edu

echo "=========================================="
echo "GPU Benchmark Job Starting"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load conda
module load anaconda3/2024.10-1
conda activate rirl

# Check environment
echo ""
echo "Environment Check:"
echo "Attempting to detect GPU..."

# Try GPU first, fall back to CPU if needed
if python -c "import jax; jax.devices('gpu')" 2>/dev/null; then
    echo "✓ GPU detected - running GPU benchmark"
    python benchmark_gpu.py
else
    echo "⚠️  GPU initialization failed - running on CPU"
    echo "Note: Results will be CPU-based"
    JAX_PLATFORMS=cpu python benchmark_gpu.py
fi

# Run the GPU benchmark
echo ""
echo "=========================================="
echo "Running GPU Benchmark"
echo "=========================================="

echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="

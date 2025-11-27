#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:10:00
#SBATCH --output=unit_test_%j.out
#SBATCH --error=unit_test_%j.err

echo "=== Running MuJoCo Dynamics Unit Test (CPU mode) ==="
echo "Node: $SLURM_NODELIST"
echo ""

module load anaconda3/2024.10-1
source activate rirl

# Configure JAX for CPU only (even though on GPU node)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu

echo "Running unit_test.py on CPU..."
python unit_test.py

echo ""
echo "=== Unit Test Complete ==="

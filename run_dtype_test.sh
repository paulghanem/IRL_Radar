#!/bin/bash
#SBATCH --job-name=dtype_test
#SBATCH --output=dtype_test_%j.out
#SBATCH --error=dtype_test_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 00:15:00

echo "==========================================================="
echo "  DTYPE Test: horizon=25, num_traj=500"
echo "==========================================================="
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

module load anaconda3/2024.10-1
module load cuda/12.4.0
source activate rirl

# Configure CUDA library paths
NVIDIA_SITE_PACKAGES=$(python -c 'import sys, os; paths = [p for p in sys.path if "site-packages" in p and os.path.exists(os.path.join(p, "nvidia"))]; print(paths[0] if paths else "")')
if [ -n "$NVIDIA_SITE_PACKAGES" ]; then
    NVIDIA_LIB_PATHS=$(find "$NVIDIA_SITE_PACKAGES/nvidia" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

python test_dtype_h25_n500.py

echo ""
echo "==========================================================="
echo "Test complete!"
echo "==========================================================="

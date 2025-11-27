#!/bin/bash
#SBATCH --job-name=h25_n125
#SBATCH --output=benchmark_h25_n125_%j.out
#SBATCH --error=benchmark_h25_n125_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 00:30:00

echo "==========================================================="
echo "  Benchmark: horizon=25, num_traj=125"
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

python main_lax.py --horizon=25 --N_steps=100 --gym_env=Walker2d --lr=1e-4 --num_traj=125 --reward_fn_updates=15 --lambda_=0.01 --rirl_iterations=1 --UB --no-save_images

echo ""
echo "==========================================================="
echo "Benchmark complete!"
echo "==========================================================="

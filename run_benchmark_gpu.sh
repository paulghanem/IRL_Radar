#!/bin/bash
#SBATCH --job-name=benchmark_loop_vs_lax
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 00:30:00

echo "=================================================="
echo "  Benchmark: Loop vs LAX on GPU"
echo "=================================================="
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl

# Configure CUDA library paths
NVIDIA_SITE_PACKAGES=$(python -c 'import sys, os; paths = [p for p in sys.path if "site-packages" in p and os.path.exists(os.path.join(p, "nvidia"))]; print(paths[0] if paths else "")')
if [ -n "$NVIDIA_SITE_PACKAGES" ]; then
    NVIDIA_LIB_PATHS=$(find "$NVIDIA_SITE_PACKAGES/nvidia" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "Running benchmark with parameters:"
echo "  N_steps: 100"
echo "  horizon: 50"
echo "  num_samples: 500"
echo "=================================================="
echo ""

python main_with_timing.py \
    --horizon=50 \
    --N_steps=100 \
    --gym_env=Walker2d \
    --lr=1e-4 \
    --num_traj=500 \
    --reward_fn_updates=1 \
    --lambda_=0.01 \
    --rirl_iterations=1 \
    --UB \
    --no-save_images \
    --seed=123

echo ""
echo "=================================================="
echo "Benchmark complete!"
echo "=================================================="

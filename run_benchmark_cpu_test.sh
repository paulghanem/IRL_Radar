#!/bin/bash
#SBATCH --job-name=benchmark_cpu_test
#SBATCH --output=benchmark_cpu_test_%j.out
#SBATCH --error=benchmark_cpu_test_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -t 00:15:00

echo "=================================================="
echo "  CPU Test: Loop vs LAX (Small Parameters)"
echo "=================================================="
echo "Node: $SLURM_NODELIST"
echo ""

module load anaconda3/2024.10-1
conda activate rirl

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Running CPU test with small parameters:"
echo "  N_steps: 5"
echo "  horizon: 5"
echo "  num_samples: 5"
echo "=================================================="
echo ""

python main_with_timing.py \
    --horizon=5 \
    --N_steps=5 \
    --gym_env=Walker2d \
    --lr=1e-4 \
    --num_traj=5 \
    --reward_fn_updates=1 \
    --lambda_=0.01 \
    --rirl_iterations=1 \
    --UB \
    --no-save_images \
    --seed=123

echo ""
echo "=================================================="
echo "CPU test complete!"
echo "=================================================="

#!/bin/bash
#SBATCH --job-name=full_train
#SBATCH --output=full_training_seed%a_%j.out
#SBATCH --error=full_training_seed%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=04:00:00
#SBATCH --account=cis250114p
#SBATCH --array=123-126

echo "=========================================="
echo "Full Training: Simplified Walker2d"
echo "=========================================="
echo "Seed: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"

# Load modules
module load anaconda3
source activate rirl

# Set environment for GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"

# Navigate to project directory
cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

# Show GPU info
nvidia-smi

echo ""
echo "Running main_lax_simple.py with GCL method..."
echo "Parameters:"
echo "  - Method: GCL (default)"
echo "  - Seed: $SLURM_ARRAY_TASK_ID"
echo "  - Iterations: 1000"
echo "  - N_steps: 1000"
echo "  - N_steps_expert: 1000"
echo "  - Reward updates: 15"
echo "  - Horizon: 50"
echo "  - Num trajectories: 2000"
echo "  - Lambda: 0.01"
echo "  - Learning rate: 1e-4"
echo "  - Q: 1e-4"
echo "  - P: 0.01"
echo "  - Hidden dim: 64"
echo "  - Sigma: 1.0"
echo ""

python3 main_lax_simple.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --gym_env SimpleWalker2d \
    --rirl_iterations 1000 \
    --N_steps 1000 \
    --N_steps_expert 1000 \
    --reward_fn_updates 15 \
    --horizon 50 \
    --num_traj 2000 \
    --lambda_ 0.01 \
    --lr 1e-4 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 64 \
    --sigma 1.0 \
    --experiment_name full_training_gcl \
    --no-save_images

echo ""
echo "End time: $(date)"
echo "=========================================="

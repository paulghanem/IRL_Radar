#!/bin/bash
#SBATCH --job-name=test_reward
#SBATCH --output=test_reward_validation_%j.out
#SBATCH --error=test_reward_validation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=00:30:00
#SBATCH --account=cis250114p

echo "=========================================="
echo "Test: Reward Validation for SimpleWalker2d"
echo "=========================================="
echo "Seed: 123"
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
echo "Running test with:"
echo "  - Iterations: 1 (just test initial trajectory)"
echo "  - N_steps: 1000"
echo "  - Seed: 123"
echo "  - Sigma: 1.0"
echo ""

python3 main_lax_simple.py \
    --seed 123 \
    --gym_env SimpleWalker2d \
    --rirl_iterations 1 \
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
    --experiment_name test_reward_validation \
    --no-save_images

echo ""
echo "End time: $(date)"
echo "=========================================="

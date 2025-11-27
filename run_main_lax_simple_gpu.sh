#!/bin/bash
#SBATCH --job-name=lax_simple
#SBATCH --output=lax_simple_%j.out
#SBATCH --error=lax_simple_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=04:00:00
#SBATCH --account=cis250114p

echo "=========================================="
echo "Main LAX with Simplified Walker2d on GPU"
echo "=========================================="
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
echo "  - Seed: 123"
echo "  - Iterations: 10"
echo "  - N_steps: 100"
echo "  - Horizon: 50"
echo "  - Num trajectories: 500"
echo "  - Lambda: 0.01"
echo "  - Hidden dim: 16"
echo ""

python main_lax_simple.py \
    --seed 123 \
    --gym_env SimpleWalker2d \
    --rirl_iterations 10 \
    --N_steps 100 \
    --N_steps_expert 100 \
    --reward_fn_updates 15 \
    --horizon 50 \
    --num_traj 500 \
    --lambda_ 0.01 \
    --lr 1e-4 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 16 \
    --sigma 1.0 \
    --experiment_name gcl_simple_walker2d \
    --no-save_images

echo ""
echo "End time: $(date)"
echo "=========================================="

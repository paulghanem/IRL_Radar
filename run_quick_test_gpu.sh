#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=00:30:00
#SBATCH --account=cis250114p

echo "=========================================="
echo "Quick Test: Simplified Walker2d on GPU"
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
echo "Running quick test (100 timesteps, 10 iterations)..."
python test_simple_walker_quick.py \
    --seed 123 \
    --horizon 50 \
    --num_traj 500 \
    --N_steps 100 \
    --rirl_iterations 10 \
    --reward_fn_updates 15 \
    --lr 1e-4 \
    --lambda_ 0.01 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 16

echo ""
echo "End time: $(date)"
echo "=========================================="

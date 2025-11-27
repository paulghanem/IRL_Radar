#!/bin/bash
#SBATCH --job-name=simple_walker_gpu
#SBATCH --output=simple_walker_gpu_%j.out
#SBATCH --error=simple_walker_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=02:00:00
#SBATCH --account=cis250114p

# Load modules
module load anaconda3

# Activate conda environment
source activate rirl

# Set JAX to use GPU
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"

# Navigate to project directory
cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

# Run the simple walker test on GPU
echo "=========================================="
echo "Running Simplified Walker2d on GPU"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Check GPU availability
nvidia-smi

echo ""
echo "Running GCL method..."
python test_simple_walker2d.py \
    --platform cuda \
    --method gcl \
    --gym_env SimpleWalker2d \
    --seed 123 \
    --horizon 50 \
    --num_traj 500 \
    --N_steps 1000 \
    --rirl_iterations 1000 \
    --reward_fn_updates 15 \
    --lr 1e-4 \
    --lambda_ 0.01 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 16

echo ""
echo "End time: $(date)"
echo "=========================================="

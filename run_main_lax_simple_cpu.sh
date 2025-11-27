#!/bin/bash
#SBATCH --job-name=lax_cpu
#SBATCH --output=lax_simple_cpu_%j.out
#SBATCH --error=lax_simple_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=RM-small
#SBATCH --time=01:00:00

echo "=========================================="
echo "Main LAX with Simplified Walker2d on CPU"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"

# Load modules
module load anaconda3
source activate rirl

# Navigate to project directory
cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

echo ""
echo "Running main_lax_simple.py with GCL method on CPU..."
echo "Parameters:"
echo "  - Method: GCL (default)"
echo "  - Seed: 123"
echo "  - Iterations: 2"
echo "  - N_steps: 50"
echo "  - Horizon: 20"
echo "  - Num trajectories: 100"
echo "  - Lambda: 0.01"
echo "  - Hidden dim: 8"
echo ""

python3 main_lax_simple.py \
    --seed 123 \
    --gym_env SimpleWalker2d \
    --rirl_iterations 2 \
    --N_steps 50 \
    --N_steps_expert 50 \
    --reward_fn_updates 5 \
    --horizon 20 \
    --num_traj 100 \
    --lambda_ 0.01 \
    --lr 1e-4 \
    --Q 1e-4 \
    --P 0.01 \
    --hidden_dim 8 \
    --sigma 1.0 \
    --experiment_name test_cpu \
    --no-save_images

echo ""
echo "End time: $(date)"
echo "=========================================="

#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:30:00
#SBATCH --output=cartpole_test_%j.out
#SBATCH --error=cartpole_test_%j.err

echo "=== CartPole GCL and RGCL Test ==="
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl
echo "✓ Environment activated"
echo ""

# Configure CUDA library paths for JAX
NVIDIA_SITE_PACKAGES=$(python -c "import sys, os; paths = [p for p in sys.path if 'site-packages' in p and os.path.exists(os.path.join(p, 'nvidia'))]; print(paths[0] if paths else '')")

if [ -n "$NVIDIA_SITE_PACKAGES" ]; then
    NVIDIA_LIB_PATHS=$(find "$NVIDIA_SITE_PACKAGES/nvidia" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "=========================================="
echo "Running GCL on CartPole-v1 (Q=1e-5)"
echo "=========================================="
python main_lax.py \
    --gym_env=CartPole-v1 \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=1000 \
    --N_steps_expert=1000 \
    --rirl_iterations=20 \
    --reward_fn_updates=15 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=1e-5 \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

echo ""
echo "=========================================="
echo "Running RGCL on CartPole-v1 (Q=1e-5)"
echo "=========================================="
python main_lax.py \
    --gym_env=CartPole-v1 \
    --seed=123 \
    --horizon=50 \
    --num_traj=500 \
    --N_steps=1000 \
    --N_steps_expert=1000 \
    --rirl_iterations=20 \
    --reward_fn_updates=15 \
    --rgcl \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=1e-5 \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

echo ""
echo "=========================================="
echo "CartPole Tests Complete"

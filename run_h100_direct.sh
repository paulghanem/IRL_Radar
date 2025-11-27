#!/bin/bash
# Direct srun execution on H100

echo "========================================"
echo "Attempting to run Walker2d on H100 via srun"
echo "========================================"

srun -N 1 -p GPU-shared --gpus=h100-80:1 -t 01:00:00 bash << 'EOFSRUN'
# Load environment
module load anaconda3/2024.10-1
source activate rirl

echo "Running on node: $SLURM_NODELIST"
echo "CUDA devices:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big

echo "Starting Walker2d experiment at: $(date)"
START=$(date +%s)

python main.py \
    --gym_env=Walker2d \
    --num_traj=500 \
    --horizon=50 \
    --N_steps=100 \
    --N_steps_expert=100 \
    --rirl_iterations=1 \
    --reward_fn_updates=15 \
    --UB \
    --seed=123 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=1e-4 \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

END=$(date +%s)
ELAPSED=$((END - START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Total time: ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"
echo "========================================"
EOFSRUN

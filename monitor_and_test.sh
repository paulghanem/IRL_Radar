#!/bin/bash
# Monitor for first Walker2d checkpoint and run test

CHECKPOINT_FILE="expert_agents/Walker2d/PPO_200000_steps.zip"

echo "Monitoring for first Walker2d checkpoint: $CHECKPOINT_FILE"
echo "Will run test with N_steps=5, horizon=5, num_traj=5 once checkpoint appears"
echo ""

# Wait for checkpoint file to appear
while [ ! -f "$CHECKPOINT_FILE" ]; do
    sleep 30  # Check every 30 seconds
    echo "$(date '+%H:%M:%S') - Waiting for checkpoint..."
done

echo ""
echo "=========================================="
echo "Checkpoint detected at $(date)"
echo "=========================================="
echo ""

# Run the test experiment
echo "Running Walker2d test on CPU with first checkpoint..."
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python main.py \
    --horizon=5 \
    --N_steps=5 \
    --gym_env=Walker2d \
    --lr=1e-4 \
    --num_traj=5 \
    --reward_fn_updates=15 \
    --lambda_=0.01 \
    --rirl_iterations=1 \
    --UB \
    --no-save_images

echo ""
echo "=========================================="
echo "Test complete at $(date)"
echo "=========================================="

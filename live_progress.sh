#!/bin/bash
# Live progress monitor for full training runs

echo "=========================================="
echo "LIVE TRAINING PROGRESS"
echo "=========================================="
date
echo ""

# Job status
echo "Job Status:"
squeue -u pghanem --name=full_train -o "%.18i %.12j %.8T %.10M %.6D %N"
echo ""
echo "=========================================="

for seed in 123 124 125 126; do
    outfile=$(ls -t full_training_seed${seed}_*.out 2>/dev/null | head -1)
    if [ -f "$outfile" ]; then
        echo ""
        echo "=== SEED $seed ==="

        # Check if training started
        if grep -q "RIRL ITERATION" "$outfile"; then
            # Get current iteration
            current_iter=$(grep "RIRL ITERATION" "$outfile" | tail -1 | awk '{print $3}')
            total_iter=$(grep "RIRL ITERATION" "$outfile" | tail -1 | awk '{print $5}')

            echo "Progress: Iteration $current_iter / $total_iter"

            # Get last 3 iteration rewards
            echo "Last 3 iterations:"
            tail -200 "$outfile" | grep "Total reward:" | tail -3 | awk '{printf "  Iter: reward = %s\n", $3}'

            # Get latest average
            last_avg=$(tail -100 "$outfile" | grep "Average reward (last" | tail -1 | awk '{print $6}')
            if [ ! -z "$last_avg" ]; then
                echo "  Current avg (last N): $last_avg"
            fi
        else
            # Still initializing
            if grep -q "Expert reward:" "$outfile"; then
                expert=$(grep "Expert reward:" "$outfile" | head -1 | awk '{print $3}')
                echo "Status: Expert generated (reward=$expert), starting training..."
            elif grep -q "Generating expert" "$outfile"; then
                echo "Status: Generating expert demonstrations..."
            elif grep -q "JAX Devices" "$outfile"; then
                echo "Status: Initializing JAX..."
            else
                echo "Status: Starting up..."
            fi
        fi
    else
        echo ""
        echo "=== SEED $seed ==="
        echo "Status: Waiting for output file..."
    fi
done

echo ""
echo "=========================================="
echo "Run this script periodically to see progress"
echo "Or use: watch -n 60 ./live_progress.sh"
echo "=========================================="

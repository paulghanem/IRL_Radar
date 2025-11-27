#!/bin/bash

echo "Monitoring full training jobs..."
echo "================================"

# Check job status
echo ""
echo "Job Status:"
squeue -u pghanem --name=full_train -o "%.18i %.12j %.8T %.10M %.6D %N"

echo ""
echo "Checking output files (last 50 lines of each seed):"
echo ""

for seed in 123 124 125 126; do
    outfile=$(ls -t full_training_seed${seed}_*.out 2>/dev/null | head -1)
    if [ -f "$outfile" ]; then
        echo "=== SEED $seed ==="
        tail -50 "$outfile" | grep -A5 "RIRL ITERATION" | tail -20
        echo ""
    fi
done

#!/bin/bash
# Summarize results from all full training runs

echo "=========================================="
echo "FULL TRAINING RESULTS SUMMARY"
echo "=========================================="
echo "Generated at: $(date)"
echo ""

expert_reward=210.76

echo "Configuration:"
echo "  - Iterations: 1000"
echo "  - N_steps: 1000"
echo "  - Horizon: 50"
echo "  - Num trajectories: 2000"
echo "  - Lambda: 0.01"
echo "  - Learning rate: 1e-4"
echo "  - Hidden dim: 64"
echo "  - Sigma: 0.0"
echo "  - Expert reward: $expert_reward"
echo ""
echo "=========================================="

for seed in 123 124 125 126; do
    outfile=$(ls -t full_training_seed${seed}_*.out 2>/dev/null | head -1)
    if [ -f "$outfile" ]; then
        echo ""
        echo "=== SEED $seed ==="

        # Extract final reward
        final_reward=$(grep "Final reward:" "$outfile" | tail -1 | awk '{print $3}')

        # Extract last iteration number
        last_iter=$(grep "RIRL ITERATION" "$outfile" | tail -1 | awk '{print $3}')

        # Extract last 10 iteration average
        last_avg=$(grep "Average reward (last 10 iter):" "$outfile" | tail -1 | awk '{print $6}')

        if [ ! -z "$final_reward" ]; then
            percentage=$(echo "scale=2; $final_reward / $expert_reward * 100" | bc)
            echo "  Final reward: $final_reward (${percentage}% of expert)"
        fi

        if [ ! -z "$last_iter" ]; then
            echo "  Iterations completed: $last_iter / 1000"
        fi

        if [ ! -z "$last_avg" ]; then
            avg_percentage=$(echo "scale=2; $last_avg / $expert_reward * 100" | bc)
            echo "  Average (last 10): $last_avg (${avg_percentage}% of expert)"
        fi

        # Show last few iterations
        echo ""
        echo "  Last 3 iterations:"
        tail -100 "$outfile" | grep -B2 "Average reward (last" | tail -9 | grep -E "(Total reward|Average reward \(last)"
    else
        echo ""
        echo "=== SEED $seed ==="
        echo "  Output file not found or job not started yet"
    fi
done

echo ""
echo "=========================================="
echo "Cross-seed statistics (when all complete):"
echo "=========================================="

# Calculate mean and std if all files exist
all_rewards=""
for seed in 123 124 125 126; do
    outfile=$(ls -t full_training_seed${seed}_*.out 2>/dev/null | head -1)
    if [ -f "$outfile" ]; then
        reward=$(grep "Final reward:" "$outfile" | tail -1 | awk '{print $3}')
        if [ ! -z "$reward" ]; then
            all_rewards="$all_rewards $reward"
        fi
    fi
done

if [ ! -z "$all_rewards" ]; then
    python3 << EOF
import numpy as np
rewards = np.array([float(x) for x in "$all_rewards".split()])
if len(rewards) > 0:
    mean = np.mean(rewards)
    std = np.std(rewards)
    expert = $expert_reward
    print(f"  Mean final reward: {mean:.4f} ± {std:.4f}")
    print(f"  Performance: {mean/expert*100:.2f}% of expert")
    print(f"  Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}")
EOF
fi

echo ""
echo "=========================================="

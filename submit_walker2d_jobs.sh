#!/bin/bash

echo "Submitting Walker2d experiments..."
echo "Seeds: 123, 124, 125, 126"
echo "Methods: UB, gail, airl, gcl, sqil"
echo "Total jobs: 20 (4 seeds × 5 methods)"
echo ""

# Array of methods
METHODS=("UB" "gail" "airl" "gcl" "sqil")

# Loop over seeds and methods
for seed in 123 124 125 126
do
    for method in "${METHODS[@]}"
    do
        echo "Submitting: Method=${method}, Seed=${seed}"
        sbatch --export=ALL,SEED=${seed},METHOD=${method} --job-name=${method}_s${seed} gpu_sbatch_walker2d
    done
done

echo ""
echo "All jobs submitted!"
echo "Use 'squeue -u $USER' to check job status"

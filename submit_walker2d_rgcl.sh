#!/bin/bash

echo "Submitting Walker2d RGCL experiments (horizon=5)..."
echo "Seeds: 123, 124, 125, 126"
echo "Method: RGCL with horizon=5"
echo "Total jobs: 4"
echo ""

# Loop over seeds
for seed in 123 124 125 126
do
    echo "Submitting: Method=RGCL, Seed=${seed}, Horizon=5"
    sbatch --export=ALL,SEED=${seed} --job-name=rgcl_h5_s${seed} gpu_sbatch_walker2d_rgcl
done

echo ""
echo "All RGCL jobs submitted!"
echo "Use 'squeue -u $USER' to check job status"

#!/bin/bash

# Submit RGCL jobs with two different Q values for comparison
SEEDS=(123 124 125 126)
ENVS=("Walker2d" "Hopper" "HalfCheetah-v4" "Swimmer")
Q_VALUES=("1e-6" "1e-4")

# Create logs directory
mkdir -p slurm_logs

# Loop through Q values, environments, and seeds
for Q_VALUE in "${Q_VALUES[@]}"; do
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            METHOD_NAME="rgcl"
            METHOD_FLAG="--rgcl"

            # Create job name with Q value to distinguish them
            JOB_NAME="${env}_${METHOD_NAME}_Q${Q_VALUE}_${seed}"

            echo "Submitting job: $JOB_NAME (V100 16GB, Q=${Q_VALUE})"

            # Submit to SLURM
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-16:1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --mail-user=ghanem.p@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Activate conda environment
module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl

echo "Running ${JOB_NAME} on \$(hostname) with Q=${Q_VALUE}"
echo "GPU Allocated: \$CUDA_VISIBLE_DEVICES"

python main.py \
    --gym_env="${env}" \
    --num_traj=500 \
    --horizon=20 \
    --N_steps=1000 \
    --N_steps_expert=1000 \
    --rirl_iterations=1000 \
    --reward_fn_updates=15 \
    ${METHOD_FLAG} \
    --seed=${seed} \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=${Q_VALUE} \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

echo "Job ${JOB_NAME} complete"
EOT

        done
    done
done

echo ""
echo "Submitted $(( ${#Q_VALUES[@]} * ${#ENVS[@]} * ${#SEEDS[@]} )) RGCL jobs total"
echo "Q values tested: ${Q_VALUES[@]}"

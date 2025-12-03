#!/bin/bash

# Submit only RGCL jobs with Q=1e-4
SEEDS=(123 124 125 126)
ENVS=("Walker2d" "Hopper" "HalfCheetah-v4" "Swimmer")

# Create logs directory
mkdir -p slurm_logs

# Loop through all combinations (RGCL only)
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        METHOD_NAME="rgcl"
        METHOD_FLAG="--rgcl"
        Q_VALUE="1e-4"

        JOB_NAME="${env}_${METHOD_NAME}_${seed}"

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

echo "Running ${JOB_NAME} on \$(hostname)"
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

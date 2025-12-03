#!/bin/bash

# Submit RGCL jobs with both Q=1e-6 and Q=1e-5 for comparison

SEEDS=(123 124 125 126)
ENVS=("Walker2d" "Hopper" "HalfCheetah-v4" "Swimmer")
Q_VALUES=("1e-6" "1e-5")

mkdir -p slurm_logs

echo "==========================================="
echo "Submitting RGCL with both Q values"
echo "==========================================="
echo ""

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for Q_VALUE in "${Q_VALUES[@]}"; do

            # Create unique job name with Q value
            Q_LABEL=$(echo $Q_VALUE | sed 's/e-/e-/')
            JOB_NAME="${env}_rgcl_Q${Q_LABEL}_${seed}"

            echo "Submitting: $JOB_NAME (Any GPU, Q=${Q_VALUE})"

            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --mail-user=ghanem.p@northeastern.edu
#SBATCH --mail-type=END,FAIL

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
    --rgcl \
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
echo "==========================================="
echo "Submission Summary:"
echo "  Environments: 4 (Walker2d, Hopper, HalfCheetah-v4, Swimmer)"
echo "  Seeds: 4 (123-126)"
echo "  Q values: 2 (1e-6, 1e-5)"
echo "  Total RGCL jobs: 32"
echo "  GPU: Any available"
echo "==========================================="

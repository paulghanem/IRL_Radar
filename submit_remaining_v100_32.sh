#!/bin/bash

# Submit only the jobs that were pending (not currently running)
# Running jobs (Walker2d and Hopper non-RGCL) will stay on V100-16

SEEDS=(123 124 125 126)
METHODS=("--rgcl" "--UB" "--airl" "--gail" "--sqil" "")

# Create logs directory
mkdir -p slurm_logs

echo "=========================================="
echo "Submitting pending jobs to V100-32"
echo "=========================================="
echo ""

# 1. Submit ALL RGCL jobs (all environments, all seeds) with Q=1e-6
echo "=== SUBMITTING ALL RGCL JOBS ==="
for env in "Walker2d" "Hopper" "HalfCheetah-v4" "Swimmer"; do
    for seed in "${SEEDS[@]}"; do
        METHOD_NAME="rgcl"
        METHOD_FLAG="--rgcl"
        Q_VALUE="1e-6"
        JOB_NAME="${env}_${METHOD_NAME}_${seed}"

        echo "Submitting: $JOB_NAME (V100-32, Q=${Q_VALUE})"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-32:1
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

echo ""
echo "=== SUBMITTING HALFCHEETAH-V4 JOBS ==="
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
        fi

        # Skip RGCL (already submitted above)
        if [ "$METHOD_NAME" == "rgcl" ]; then
            continue
        fi

        Q_VALUE="1e-5"
        JOB_NAME="HalfCheetah-v4_${METHOD_NAME}_${seed}"

        echo "Submitting: $JOB_NAME (V100-32, Q=${Q_VALUE})"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --mail-user=ghanem.p@northeastern.edu
#SBATCH --mail-type=END,FAIL

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl

echo "Running ${JOB_NAME} on \$(hostname)"
echo "GPU Allocated: \$CUDA_VISIBLE_DEVICES"

python main.py \
    --gym_env="HalfCheetah-v4" \
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

echo ""
echo "=== SUBMITTING SWIMMER JOBS ==="
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
        fi

        # Skip RGCL (already submitted above)
        if [ "$METHOD_NAME" == "rgcl" ]; then
            continue
        fi

        Q_VALUE="1e-5"
        JOB_NAME="Swimmer_${METHOD_NAME}_${seed}"

        echo "Submitting: $JOB_NAME (V100-32, Q=${Q_VALUE})"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --mail-user=ghanem.p@northeastern.edu
#SBATCH --mail-type=END,FAIL

module load anaconda3/2024.10-1
module load cuda/12.4.0
conda activate rirl

echo "Running ${JOB_NAME} on \$(hostname)"
echo "GPU Allocated: \$CUDA_VISIBLE_DEVICES"

python main.py \
    --gym_env="Swimmer" \
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

echo ""
echo "=========================================="
echo "Summary:"
echo "  RGCL jobs (all envs): 16"
echo "  HalfCheetah-v4 jobs: 20"
echo "  Swimmer jobs: 20"
echo "  Total submitted: 56"
echo ""
echo "Running jobs on V100-16: ~43 (will continue)"
echo "=========================================="

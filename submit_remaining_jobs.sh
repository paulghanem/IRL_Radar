#!/bin/bash

# Submit remaining HalfCheetah and all Swimmer jobs to any available GPU

SEEDS=(124 125 126)  # Remaining seeds for HalfCheetah
SWIMMER_SEEDS=(123 124 125 126)  # All seeds for Swimmer

mkdir -p slurm_logs

echo "==========================================="
echo "Submitting remaining jobs (Any GPU)"
echo "==========================================="
echo ""

# HalfCheetah seed 124 - remaining methods (gail, sqil, baseline)
for method in "--gail" "--sqil" ""; do
    if [ -z "$method" ]; then
        METHOD_NAME="baseline"
        METHOD_FLAG=""
        Q_VALUE="1e-5"
    else
        METHOD_NAME="${method//-/}"
        METHOD_FLAG="$method"
        Q_VALUE="1e-5"
    fi

    JOB_NAME="HalfCheetah-v4_${METHOD_NAME}_124"
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
    --gym_env="HalfCheetah-v4" \
    --num_traj=500 \
    --horizon=20 \
    --N_steps=1000 \
    --N_steps_expert=1000 \
    --rirl_iterations=1000 \
    --reward_fn_updates=15 \
    ${METHOD_FLAG} \
    --seed=124 \
    --lr=1e-4 \
    --lambda_=0.01 \
    --Q=${Q_VALUE} \
    --P=1e-2 \
    --hidden_dim=16 \
    --no-save_images

echo "Job ${JOB_NAME} complete"
EOT
done

# HalfCheetah seeds 125 and 126 - all methods
for seed in 125 126; do
    for method in "--rgcl" "--UB" "--airl" "--gail" "--sqil" ""; do
        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
            Q_VALUE="1e-5"
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
            if [ "$METHOD_NAME" == "rgcl" ]; then
                Q_VALUE="1e-6"
            else
                Q_VALUE="1e-5"
            fi
        fi

        JOB_NAME="HalfCheetah-v4_${METHOD_NAME}_${seed}"
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

# All Swimmer jobs - all seeds and methods
for seed in "${SWIMMER_SEEDS[@]}"; do
    for method in "--rgcl" "--UB" "--airl" "--gail" "--sqil" ""; do
        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
            Q_VALUE="1e-5"
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
            if [ "$METHOD_NAME" == "rgcl" ]; then
                Q_VALUE="1e-6"
            else
                Q_VALUE="1e-5"
            fi
        fi

        JOB_NAME="Swimmer_${METHOD_NAME}_${seed}"
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
echo "==========================================="
echo "Submission Summary:"
echo "  HalfCheetah remaining: 15 jobs"
echo "  Swimmer: 24 jobs"
echo "  Total submitted: 39 jobs"
echo "  GPU: Any available"
echo "==========================================="

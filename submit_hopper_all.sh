#!/bin/bash

# Submit ALL Hopper experiments
# RGCL with both Q=1e-6 and Q=1e-5
# Other methods with Q=1e-5

SEEDS=(123 124 125 126)
METHODS=(("--rgcl" "1e-6") ("--rgcl" "1e-5") ("--UB" "1e-5") ("--airl" "1e-5") ("--gail" "1e-5") ("--sqil" "1e-5") ("" "1e-5"))

mkdir -p slurm_logs

echo "==========================================="
echo "Submitting ALL Hopper experiments"
echo "==========================================="
echo ""

# RGCL with both Q values
for seed in "${SEEDS[@]}"; do
    for Q_VALUE in "1e-6" "1e-5"; do
        Q_LABEL=$(echo $Q_VALUE | sed 's/e-/e-/')
        JOB_NAME="Hopper_rgcl_Q${Q_LABEL}_${seed}"

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
    --gym_env="Hopper" \
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

# Other methods (UB, AIRL, GAIL, SQIL, baseline)
for seed in "${SEEDS[@]}"; do
    for method in "--UB" "--airl" "--gail" "--sqil" ""; do

        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
        fi

        Q_VALUE="1e-5"
        JOB_NAME="Hopper_${METHOD_NAME}_${seed}"

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
    --gym_env="Hopper" \
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
echo "  RGCL: 8 jobs (4 seeds × 2 Q values)"
echo "  Other methods: 20 jobs (5 methods × 4 seeds)"
echo "  Total Hopper jobs: 28"
echo "  GPU: Any available"
echo "==========================================="

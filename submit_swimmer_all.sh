#!/bin/bash

# Submit ALL Swimmer jobs for all methods and seeds on V100-32
# With the fixed case-sensitive swimmer.xml filename

SEEDS=(123 124 125 126)
METHODS=("--rgcl" "--UB" "--airl" "--gail" "--sqil" "")

mkdir -p slurm_logs

echo "=========================================="
echo "Submitting ALL Swimmer jobs (V100-32)"
echo "=========================================="
echo ""

for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do

        if [ -z "$method" ]; then
            METHOD_NAME="baseline"
            METHOD_FLAG=""
        else
            METHOD_NAME="${method//-/}"
            METHOD_FLAG="$method"
        fi

        # Set Q value based on method
        if [ "$METHOD_NAME" == "rgcl" ]; then
            Q_VALUE="1e-6"
        else
            Q_VALUE="1e-5"
        fi

        JOB_NAME="Swimmer_${METHOD_NAME}_${seed}"

        echo "Submitting: $JOB_NAME (L40S, Q=${Q_VALUE})"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=l40s:1
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
echo "=========================================="
echo "Total Swimmer jobs submitted: 24"
echo "  Methods: rgcl, UB, airl, gail, sqil, baseline"
echo "  Seeds: 123, 124, 125, 126"
echo "  GPU: V100-32"
echo "=========================================="

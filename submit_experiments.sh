#!/bin/bash

# 1. Define your variables
SEEDS=(123 124 125 126)
ENVS=("Walker2d" "Hopper" "HalfCheetah-v4" "Swimmer")
METHODS=("--rgcl" "--UB" "--airl" "--gail" "--sqil" "")

# 2. Create a logs directory
mkdir -p slurm_logs

# 3. Loop through all combinations
for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
            
            # Logic to handle the "no flag" naming convention
            if [ -z "$method" ]; then
                METHOD_NAME="baseline"
                METHOD_FLAG=""
            else
                METHOD_NAME="${method//-/}"
                METHOD_FLAG="$method"
            fi

            JOB_NAME="${env}_${METHOD_NAME}_${seed}"

            # Set Q value based on method
            if [ "$method" == "--rgcl" ]; then
                Q_VALUE="1e-6"
            else
                Q_VALUE="1e-5"
            fi

            echo "Submitting job: $JOB_NAME (L40S, Q=${Q_VALUE})"

            # 4. Submit to SLURM
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}.out
#SBATCH --error=slurm_logs/${JOB_NAME}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
<<<<<<< Updated upstream
#SBATCH --gpus=v100-32:1
=======
#SBATCH --gpus=l40s:1
>>>>>>> Stashed changes
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
done

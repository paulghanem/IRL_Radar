#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1

#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=8:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ghanem.p@northeastern.edu


module load cuda
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh

conda activate rirl
python generate_expert_buffer.py --seed=$SEED --N_steps_expert=1000000 --gym_env=HalfCheetah-v4 --lr=1e-3 --num_traj=250 --reward_fn_updates=15 --lambda_=0.001  --rirl_iterations=1  --Q=1e-5 --P=1e-2 --hidden_dim=16  --sigma=1e-3

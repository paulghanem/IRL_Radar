# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:07:06 2025

@author: siliconsynapse
"""
#!/bin/bash


# List of methods, include an empty string as ""
for method in airl rgcl gail UB sqil ""

do
  for seed in {123..134}
  do
    gpu_id=$((seed - 123))  # assign GPU 0–11

    echo "Running seed=$seed with method='$method' on GPU $gpu_id"

    # Construct the command
    CMD="CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      --seed=$seed \
      --horizon=50 \
      --N_steps=200 \
      --gym_env=Walker2d \
      --lr=1e-4 \
      --num_traj=500 \
      --reward_fn_updates=15 \
      --lambda_=0.01 \
      --rirl_iterations=100 \
      --Q=1e-4 \
      --P=1e-2 \
      --hidden_dim=16"

    # Add --method only if method is not empty
    if [[ -n "$method" ]]; then
      CMD+=" --method=$method"
    fi

    # Print and run the command
    echo "$CMD"
    eval "$CMD"

  done
done

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:07:06 2025

@author: siliconsynapse
"""
#!/bin/bash

# Function to find a free GPU with enough available memory
wait_for_gpu() {
  while true; do
    for gpu_id in {0..7}; do
      mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu_id+1))p")
      if [[ "$mem_free" -gt 8000 ]]; then  # adjust threshold as needed
        echo "Found free GPU $gpu_id with $mem_free MiB free"
        echo $gpu_id
        return
      fi
    done
    echo "No free GPU found. Waiting..."
    sleep 30
  done
}

# List of methods including empty
for method in gail UB 
do
  for seed in {123..134}
  do
    # Wait for a free GPU and get its ID
    gpu_id=$(wait_for_gpu)

    echo "Launching seed=$seed with method='$method' on GPU $gpu_id"

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

    if [[ -n "$method" ]]; then
      CMD+=" --method=$method"
    fi

    # Run the command in the background
    echo "$CMD"
    eval "$CMD" &

    # Optional: limit number of concurrent jobs
    while [[ $(jobs -r | wc -l) -ge 8 ]]; do
      sleep 10
    done
  done
done

# Wait for all background jobs to finish
wait

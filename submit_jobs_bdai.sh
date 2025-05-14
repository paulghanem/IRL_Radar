#!/bin/bash

# Function to find a free GPU with enough available memory
wait_for_gpu() {
  while true; do
    for gpu_id in {0..3}; do
      mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v id="$gpu_id" 'NR == id + 1 {print $1}')
      >&2 echo "GPU $gpu_id - Memory Free: $mem_free MiB"  # Debug output to stderr
      if [[ "$mem_free" -gt 3000 ]]; then  # Adjust threshold as needed
        >&2 echo "Found free GPU $gpu_id with $mem_free MiB free"
        echo "$gpu_id"  # Only this is captured by the caller
        return
      fi
    done
    >&2 echo "No free GPU found. Waiting..."
    sleep 30
  done
}

# Create logs directory if it doesn't exist
mkdir -p logs

# List of methods
for method in airl rgcl; do
  for seed in {123..134}; do
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
      CMD+=" --$method"
    fi

    log_file="logs/${method}_seed_${seed}.log"
    echo "Running: $CMD"
    eval "$CMD" 

    sleep 120 
  done
done

# Wait for all background jobs to finish
wait

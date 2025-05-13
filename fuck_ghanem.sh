#!/bin/bash

for task in airl rgcl gail; do
    for seed in {130..134}; do
        echo "Running task --$task with seed $seed"
        python main.py \
            --seed="$seed" \
            --horizon=50 \
            --N_steps=200 \
            --gym_env=Hopper \
            --lr=1e-4 \
            --num_traj=500 \
            --reward_fn_updates=15 \
            --lambda_=0.01 \
            --rirl_iterations=100 \
            --Q=1e-4 \
            --P=1e-2 \
            --hidden_dim=16 \
            --"$task" \
            >> "output_seed_${seed}_${task}.log" 2>&1
    done
done


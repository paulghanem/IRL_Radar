#!/bin/bash
module load anaconda3
source activate rirl
python test_simple_walker2d.py --platform=cpu --method=gcl --gym_env=SimpleWalker2d --seed=123 --horizon=50 --num_traj=500 --N_steps=10 --rirl_iterations=1 --s_dim=18 --a_dim=6

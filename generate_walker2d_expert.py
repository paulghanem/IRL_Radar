#!/usr/bin/env python3
"""
Generate expert demonstrations for Walker2d using trained PPO policy
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import numpy as np
from utils.helpers import GenerateDemo

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--N_steps', type=int, default=1000)
parser.add_argument('--gym_env', type=str, default='Walker2d')
args = parser.parse_args()

print("=" * 80)
print("GENERATING WALKER2D EXPERT DEMONSTRATIONS")
print("=" * 80)
print(f"Environment: {args.gym_env}")
print(f"Steps: {args.N_steps}")
print(f"Seed: {args.seed}")
print(f"Expert model: expert_agents/{args.gym_env}/PPO.zip")
print("=" * 80)

# Generate expert demonstrations
demo_generator = GenerateDemo(args.gym_env, max_frames=args.N_steps)
states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(args.seed)

print("\n" + "=" * 80)
print("EXPERT DEMONSTRATION GENERATED")
print("=" * 80)
print(f"States shape: {states_d.shape}")
print(f"Actions shape: {actions_d.shape}")
print(f"Total reward: {rewards_demo[-1]:.2f}")
print(f"Average reward per step: {rewards_demo[-1] / args.N_steps:.4f}")
print("=" * 80)

# Save demonstrations
fname = f"expert_demo_{args.gym_env}_seed{args.seed}_steps{args.N_steps}.npz"
np.savez(fname, states=states_d, actions=actions_d, rewards=rewards_demo)
print(f"\nSaved to: {fname}")

"""
IRL with PPO Policy - CartPole Implementation
This script uses PPO (Proximal Policy Optimization) instead of MPPI for trajectory generation
"""

from flax.training import train_state
import flax
import optax
import os
import argparse

import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jit
import time

import gymnax
from flax import struct
import pdb

from cost_jax import CostNN, apply_model, apply_model_AIRL, update_model, apply_model_SQIL
from src.control.PPO_simple import SimplePPO, PolicyModel, CriticModel
from src.control.dynamics import get_step_model
from utils.helpers import GenerateDemo

print(jax.devices())

# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo=False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = jnp.array(traj[0])
        if is_Demo:
            probs = jnp.ones((states.shape[0], 1))
        else:
            probs = jnp.array(traj[1]).reshape(-1, 1)
        actions = jnp.array(traj[2])
        x = jnp.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return jnp.array(step_list)


parser = argparse.ArgumentParser(description='IRL with PPO Policy',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Experiment settings
parser.add_argument('--seed', default=123, type=int, help='Random seed')
parser.add_argument('--N_steps', default=200, type=int, help='Number of steps per episode')
parser.add_argument('--N_steps_expert', default=200, type=int, help='Number of expert steps')
parser.add_argument('--rirl_iterations', default=100, type=int, help='Number of training iterations')
parser.add_argument('--reward_fn_updates', default=10, type=int, help='Reward function updates per iteration')
parser.add_argument('--hidden_dim', default=64, type=int, help='Hidden layer size for cost network')
parser.add_argument('--runs', default=1, type=int, help='Number of runs')

# Save settings
parser.add_argument('--results_savepath', default="results_ppo", type=str, help='Results folder')
parser.add_argument('--experiment_name', default="experiment_ppo", type=str, help='Experiment name')

# Learning rates
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for cost function')
parser.add_argument('--ppo_lr', default=3e-4, type=float, help='Learning rate for PPO policy')

# IRL method flags
parser.add_argument('--airl', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Use AIRL')
parser.add_argument('--gail', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Use GAIL')
parser.add_argument('--sqil', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Use SQIL')
parser.add_argument('--UB', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Upper bound loss')

# Environment
parser.add_argument('--gym_env', default="CartPole-v1", type=str, help='Gym environment')

# PPO settings
parser.add_argument('--rollout_length', default=200, type=int, help='PPO rollout length')
parser.add_argument('--ppo_epochs', default=10, type=int, help='PPO update epochs')
parser.add_argument('--ppo_batch_size', default=64, type=int, help='PPO batch size')
parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
parser.add_argument('--clip_eps', default=0.2, type=float, help='PPO clip epsilon')

args = parser.parse_args()

# Setup method name
args.airl = True if args.gail else args.airl

print("Using AIRL:", args.airl)
print("Using GAIL:", args.gail)

if args.airl and not args.gail:
    method = "airl-ppo"
elif args.gail:
    method = "gail-ppo"
elif args.UB:
    method = "UB-ppo"
elif args.sqil:
    method = "sqil-ppo"
else:
    method = "gcl-ppo"

# Setup save paths
args.results_savepath = os.path.join(args.results_savepath, args.experiment_name) + f"_{args.seed}"
os.makedirs(args.results_savepath, exist_ok=True)

from datetime import datetime
from pytz import timezone
import json

tz = timezone('EST')
print("Experiment Start @", datetime.now(tz))
print("Experiment Saved @", args.results_savepath)

# Save hyperparameters
with open(os.path.join(args.results_savepath, "hyperparameters.json"), "w") as outfile:
    json.dump(vars(args), outfile, indent=2)

# Initialize tracking
mean_rewards = []
mean_costs = []
mean_loss_rew = []

D_demo, D_samp = np.array([]), jnp.array([])

seeds = np.array([args.seed])
args.runs = np.shape(seeds)[0]
epoch_cost_runs = []
expert_cost_runs = []

for run in range(args.runs):
    args.seed = int(seeds[run])
    print(f"\n{'='*60}")
    print(f"Run {run+1}/{args.runs}, Seed: {args.seed}")
    print(f"{'='*60}\n")

    epoch_cost = []
    expert_cost = []

    for i in range(args.rirl_iterations):
        if i == 0:
            # ========== GENERATE EXPERT DEMONSTRATIONS ==========
            print("Generating expert demonstrations...")
            demo_generator = GenerateDemo(args.gym_env, max_frames=args.N_steps_expert)
            states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(args.seed)
            expert_reward_total = float(jnp.sum(rewards_demo))
            print(f"Expert reward: {expert_reward_total:.2f}")

            args.DEMO_BATCH = min(args.N_steps, states_d.shape[0])
            args.a_dim = actions_d.shape[-1]
            args.s_dim = states_d.shape[-1]

            # ========== INITIALIZE COST NETWORK ==========
            cost_f = CostNN(state_dims=args.s_dim, hidden_dim=args.hidden_dim)
            init_rng = jax.random.key(0)

            variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
            params = variables['params']
            tx = optax.adam(learning_rate=args.lr)
            state_train = train_state.TrainState.create(
                apply_fn=cost_f.apply,
                params=params,
                tx=tx
            )

            # ========== INITIALIZE PPO POLICY ==========
            print("Initializing PPO policy...")

            # Get dynamics function for the environment
            dynamics_fn = get_step_model(args.gym_env, env)

            # Create policy and value networks
            model_p = PolicyModel(action_dim=args.a_dim)
            model_v = CriticModel()

            dummy_input = jnp.zeros((1, args.s_dim))
            params_p = model_p.init(init_rng, dummy_input)['params']
            params_v = model_v.init(init_rng, dummy_input)['params']

            tx_p = optax.adam(learning_rate=args.ppo_lr)
            tx_v = optax.adam(learning_rate=args.ppo_lr)

            state_train_p = train_state.TrainState.create(
                apply_fn=model_p.apply,
                params=params_p,
                tx=tx_p
            )
            state_train_v = train_state.TrainState.create(
                apply_fn=model_v.apply,
                params=params_v,
                tx=tx_v
            )

            # Create PPO policy
            policy = SimplePPO(
                state_dim=args.s_dim,
                action_dim=args.a_dim,
                dynamics=dynamics_fn,
                policy_model=state_train_p,
                policy_net=model_p,
                args=args,
                rollout_length=args.rollout_length,
                value_fn=state_train_v
            )

            # Prepare demo data
            D_demo = np.array([])
            demo_trajs = [[states_d, actions_d, actions_d]]
            D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
            D_demo = jnp.concatenate((D_demo[:, :args.s_dim], D_demo[:, args.s_dim:]), axis=1)

        # ========== GENERATE TRAJECTORIES WITH PPO ==========
        print(f"\nIteration {i+1}/{args.rirl_iterations}")
        print("Generating trajectories with PPO...")

        start = time.time()

        # Run PPO rollout
        policy.generate_session_lax(args, D_demo, frame_skip=1, dt=0.02)

        # Get rollout data
        states, actions, rewards, dones, log_probs, next_states = policy.buffer.get()

        end = time.time()

        total_reward = jnp.sum(rewards)
        print(f"Rollout time: {end - start:.4f}s, Total reward: {total_reward:.2f}")

        # Prepare sample data for cost learning
        sample_trajs = [[states, log_probs, actions]]
        D_samp = np.array([])
        D_samp = preprocess_traj(sample_trajs, D_samp)

        # ========== UPDATE COST FUNCTION ==========
        print("Updating cost function...")
        loss_rew = []

        for update in range(args.reward_fn_updates):
            # Sample from demonstrations and agent trajectories
            selected_samp_idx = np.random.choice(len(D_samp), min(args.DEMO_BATCH, len(D_samp)), replace=False)
            selected_demo_idx = np.random.choice(len(D_demo), min(args.DEMO_BATCH, len(D_demo)), replace=False)

            D_s_samp = D_samp[selected_samp_idx]
            D_s_demo = D_demo[selected_demo_idx]

            states_samp = D_s_samp[:, :args.s_dim]
            probs_samp = D_s_samp[:, args.s_dim]
            actions_samp = D_s_samp[:, args.s_dim+1:]

            states_expert = D_s_demo[:, :args.s_dim]
            probs_experts = D_s_demo[:, args.s_dim]
            actions_expert = D_s_demo[:, args.s_dim+1:]

            # Compute gradients and loss
            if args.airl:
                grads, loss_IOC = apply_model_AIRL(
                    state_train, states_samp, actions_samp,
                    states_expert, actions_expert,
                    probs_samp, probs_experts, args.UB
                )
            elif args.sqil:
                grads, loss_IOC = apply_model_SQIL(
                    state_train, states_samp, actions_samp,
                    states_expert, actions_expert,
                    probs_samp, probs_experts
                )
            else:
                grads, loss_IOC = apply_model(
                    state_train, states_samp, actions_samp,
                    states_expert, actions_expert,
                    probs_samp, probs_experts, args.UB
                )

            state_train = update_model(state_train, grads)
            loss_rew.append(loss_IOC)

        mean_loss = np.mean(loss_rew)
        print(f"Cost function loss: {mean_loss:.6f}")

        # ========== UPDATE PPO POLICY ==========
        print("Updating PPO policy...")
        policy.update_ppo(
            states=states,
            actions=actions,
            rewards=rewards.flatten(),
            dones=dones.flatten(),
            log_probs_old=log_probs.flatten(),
            next_states=next_states,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            num_epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size
        )

        # Track progress
        epoch_cost.append(float(total_reward))
        expert_cost.append(expert_reward_total)
        mean_loss_rew.append(mean_loss)

        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            recent_reward = np.mean(epoch_cost[-10:])
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}: Avg Reward (last 10) = {recent_reward:.2f}")
            print(f"Expert Reward = {expert_reward_total:.2f}")
            print(f"{'='*60}\n")

    # Save results
    epoch_cost_runs.append(epoch_cost)
    expert_cost_runs.append(expert_cost)

# Save final results
save_dir = os.path.join("results", args.gym_env, method)
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, f"cost_seed={args.seed}.npy"), epoch_cost_runs)
np.save(os.path.join(save_dir, f"expert_cost_seed={args.seed}.npy"), expert_cost_runs)
np.save(os.path.join(save_dir, f"loss_seed={args.seed}.npy"), mean_loss_rew)

print("\n" + "="*60)
print("Training Complete!")
print(f"Results saved to: {save_dir}")
print("="*60)

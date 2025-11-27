#!/usr/bin/env python3
"""
Simple benchmark: generate_session_loop vs generate_session_lax
Step 1: Test LAX on CPU with small params
Step 2: Full benchmark on GPU with large params
"""
import os
import sys
import time
import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Small test (5 steps, 5 horizon, 5 samples, CPU)')
parser.add_argument('--gpu', action='store_true', help='Use GPU (otherwise CPU)')
args_bench = parser.parse_args()

# Set platform
if args_bench.test or not args_bench.gpu:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    platform = 'CPU'
else:
    os.environ['JAX_PLATFORMS'] = 'cuda'
    platform = 'GPU'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from argparse import Namespace

# Import project code
sys.path.insert(0, '/ocean/projects/cis250114p/pghanem/IRL_Radar_big')
from utils.helpers import GenerateDemo
from src.control.mppi_class import MPPI
from cost_jax import CostNN
from flax.training import train_state
import optax

print("=" * 80)
print(f"Benchmark: generate_session_loop vs generate_session_lax")
print("=" * 80)
print(f"Platform: {platform}")
print(f"JAX devices: {jax.devices()}")
print()

# Set parameters
if args_bench.test:
    N_STEPS, HORIZON, NUM_SAMPLES = 5, 5, 5
    print("MODE: Small Test (CPU)")
else:
    N_STEPS, HORIZON, NUM_SAMPLES = 100, 50, 500
    print(f"MODE: Full Benchmark ({platform})")

print(f"Parameters:")
print(f"  N_steps: {N_STEPS}")
print(f"  horizon: {HORIZON}")
print(f"  num_samples: {NUM_SAMPLES}")
print(f"  Environment: Walker2d")
print("=" * 80)
print()

# Create args
args = Namespace(
    seed=123, dt=0.02, frame_skip=5,
    s_dim=17, a_dim=6,
    N_steps=N_STEPS, gym_env='Walker2d', gail=False,
    horizon=HORIZON, num_traj=NUM_SAMPLES,
    lambda_=0.01, hidden_dim=128
)

print("Loading demo...")
demo_gen = GenerateDemo(args.gym_env, max_frames=1000)
states_d, actions_d, rewards_demo, env = demo_gen.generate_demo(seed=args.seed)
D_demo = jnp.concatenate([states_d, actions_d], axis=1)
print(f"  Demo shape: {D_demo.shape}")

print("Initializing MPPI and cost function...")
cost_nn = CostNN(state_dims=args.s_dim, hidden_dim=args.hidden_dim)
dummy_state = jnp.zeros((1, args.s_dim))
params = cost_nn.init(jax.random.PRNGKey(0), dummy_state)
tx = optax.adam(1e-4)
state_train = train_state.TrainState.create(
    apply_fn=cost_nn.apply,
    params=params,
    tx=tx
)

# Create MPPI controller
from src.control.mppi_class import MPPI
from utils.mujoco_dynamics import load_mjx_env

mjx_model, _ = load_mjx_env(args.gym_env)

policy = MPPI(
    state_train=state_train,
    horizon=args.horizon,
    num_samples=args.num_traj,
    dim_state=args.s_dim,
    dim_control=args.a_dim,
    dynamics=None,  # Uses MuJoCo
    cost_func=None,  # Not needed for UB
    u_min=-1.0,
    u_max=1.0,
    sigmas=jnp.ones(args.a_dim),
    lambda_=args.lambda_,
    env=env,
    mjx_model=mjx_model,
    gym_env=args.gym_env,
    dt=args.dt,
    frame_skip=args.frame_skip,
    zero_mean=True
)
print("  MPPI initialized")
print()

# Benchmark LOOP version
print("=" * 80)
print("BENCHMARKING LOOP VERSION")
print("=" * 80)
start = time.time()
trajs_loop = policy.generate_session_loop(args, state_train, D_demo)
jax.block_until_ready(trajs_loop[0])
end = time.time()
loop_time = end - start
print(f"✓ Loop completed in {loop_time:.4f} seconds")
print(f"  Return: {trajs_loop[3]}")
print()

# Benchmark LAX version
print("=" * 80)
print("BENCHMARKING LAX VERSION")
print("=" * 80)
start = time.time()
trajs_lax = policy.generate_session_lax(args, state_train, D_demo)
jax.block_until_ready(trajs_lax[0])
end = time.time()
lax_time = end - start
print(f"✓ LAX completed in {lax_time:.4f} seconds")
print(f"  Return: {trajs_lax[3]}")
print()

# Results
print("=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)
print(f"Loop version: {loop_time:.4f} seconds")
print(f"LAX version:  {lax_time:.4f} seconds")
speedup = loop_time / lax_time
print(f"Speedup: {speedup:.2f}x {'(LAX faster)' if speedup > 1 else '(Loop faster)'}")
print("=" * 80)

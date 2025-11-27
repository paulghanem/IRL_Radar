#!/usr/bin/env python3
"""
Test lax version with small parameters to verify syntax
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import jax
import jax.numpy as jnp
from argparse import Namespace

# Add src to path
sys.path.insert(0, '/ocean/projects/cis250114p/pghanem/IRL_Radar_big')

from utils.helpers import GenerateDemo
from src.control.mppi_class import MPPI

print("=" * 60)
print("Testing LAX version with small parameters on CPU")
print("=" * 60)
print(f"JAX devices: {jax.devices()}")
print()

# Create minimal args
args = Namespace(
    seed=123,
    dt=0.02,
    frame_skip=5,
    s_dim=17,
    a_dim=6,
    N_steps=5,
    gym_env='Walker2d',
    gail=False,
    horizon=5,
    num_samples=5,
    lambda_=0.01,
    exploration=1.0,
    zero_mean=True
)

print("Loading expert demonstration...")
demo_generator = GenerateDemo(args.gym_env, max_frames=1000)
states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(seed=args.seed)
D_demo = jnp.concatenate([states_d, actions_d], axis=1)
print(f"Demo shape: {D_demo.shape}")
print()

print("Initializing MPPI controller...")
mppi = MPPI(
    env_name=args.gym_env,
    horizon=args.horizon,
    num_samples=args.num_samples,
    dim_state=args.s_dim,
    dim_ctrl=args.a_dim,
    dynamics=None,
    reward_fn=None,
    u_min=-1.0,
    u_max=1.0,
    lambda_=args.lambda_,
    noise_sigma=1.0,
    noise_mu=0.0,
    exploration=args.exploration,
    zero_mean=args.zero_mean,
    dt=args.dt,
    frame_skip=args.frame_skip,
    use_mujoco=True
)
print("MPPI initialized")
print()

# Create dummy state_train (not used in UB mode but needed as parameter)
from flax.training import train_state
import optax

class DummyModel:
    def apply(self, params, x):
        return jnp.zeros(1)

dummy_params = {'dummy': jnp.zeros(1)}
dummy_tx = optax.adam(1e-4)
state_train = train_state.TrainState.create(
    apply_fn=DummyModel().apply,
    params=dummy_params,
    tx=dummy_tx
)

print("Testing generate_session_lax...")
try:
    states, probs, actions, rewards = mppi.generate_session_lax(
        args=args,
        state_train=state_train,
        D_demo=D_demo
    )
    print(f"✓ LAX version succeeded!")
    print(f"  States: {len(states)} steps")
    print(f"  Actions: {len(actions)} steps")
    print(f"  Total reward: {rewards}")
    print()
except Exception as e:
    print(f"✗ LAX version failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("LAX version syntax test PASSED!")
print("=" * 60)

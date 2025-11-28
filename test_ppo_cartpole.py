"""
Simple test script to verify PPO works on CartPole
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax
import gymnax

from src.control.PPO_simple import SimplePPO, PolicyModel, CriticModel
from src.control.dynamics import get_step_model, cartpole_step
from utils.helpers import GenerateDemo

print("JAX devices:", jax.devices())

# Configuration
class Args:
    def __init__(self):
        self.seed = 42
        self.gym_env = "CartPole-v1"
        self.s_dim = 4
        self.a_dim = 1
        self.N_steps = 200
        self.ppo_lr = 3e-4
        self.rollout_length = 200

args = Args()

# Generate expert demonstrations
print("\n" + "="*60)
print("Generating expert demonstrations...")
print("="*60)

demo_generator = GenerateDemo(args.gym_env, max_frames=args.N_steps)
states_d, actions_d, rewards_demo, env = demo_generator.generate_demo(args.seed)

print(f"Expert trajectories: {states_d.shape[0]} steps")
print(f"Expert reward: {float(jnp.sum(rewards_demo)):.2f}")
print(f"State shape: {states_d.shape}")
print(f"Action shape: {actions_d.shape}")

# Prepare demo data
D_demo = jnp.concatenate([states_d, jnp.ones((states_d.shape[0], 1)), actions_d], axis=1)

# Initialize PPO
print("\n" + "="*60)
print("Initializing PPO policy...")
print("="*60)

init_rng = jax.random.key(args.seed)

# Policy network
model_p = PolicyModel(action_dim=args.a_dim)
dummy_input = jnp.zeros((1, args.s_dim))
params_p = model_p.init(init_rng, dummy_input)['params']
tx_p = optax.adam(learning_rate=args.ppo_lr)
state_train_p = train_state.TrainState.create(
    apply_fn=model_p.apply,
    params=params_p,
    tx=tx_p
)

# Value network
model_v = CriticModel()
params_v = model_v.init(init_rng, dummy_input)['params']
tx_v = optax.adam(learning_rate=args.ppo_lr)
state_train_v = train_state.TrainState.create(
    apply_fn=model_v.apply,
    params=params_v,
    tx=tx_v
)

# Create PPO policy
policy = SimplePPO(
    state_dim=args.s_dim,
    action_dim=args.a_dim,
    dynamics=cartpole_step,
    policy_model=state_train_p,
    policy_net=model_p,
    args=args,
    rollout_length=args.rollout_length,
    value_fn=state_train_v
)

print("PPO policy initialized successfully!")

# Test rollout
print("\n" + "="*60)
print("Testing PPO rollout on CartPole...")
print("="*60)

try:
    # Run a rollout
    policy.generate_session_lax(args, D_demo, frame_skip=1, dt=0.02)

    # Get the buffer data
    states, actions, rewards, dones, log_probs, next_states = policy.buffer.get()

    total_reward = jnp.sum(rewards)

    print("[OK] Rollout successful!")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward: {jnp.mean(rewards):.4f}")

    # Test PPO update
    print("\n" + "="*60)
    print("Testing PPO update...")
    print("="*60)

    policy.update_ppo(
        states=states,
        actions=actions,
        rewards=rewards.flatten(),
        dones=dones.flatten(),
        log_probs_old=log_probs.flatten(),
        next_states=next_states,
        gamma=0.99,
        clip_eps=0.2,
        num_epochs=5,
        batch_size=64
    )

    print("[OK] PPO update successful!")

    print("\n" + "="*60)
    print("All tests passed! [OK]")
    print("="*60)

except Exception as e:
    print(f"\n[ERROR] Error occurred: {e}")
    import traceback
    traceback.print_exc()

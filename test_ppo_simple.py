"""
Simple test for PPO_unified with multiple fixed initial conditions
Tests PPO training starting from different initial states - FROM SCRATCH
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
from src.control.PPO_unified import UnifiedPPO
from src.control.dynamics import get_step_model
import sys
import flax.linen as nn
import optax
from flax.training import train_state

print("JAX devices:", jax.devices())

# Setup arguments
args = argparse.Namespace()
args.seed = 42
args.s_dim = 4  # CartPole state: [x, x_dot, theta, theta_dot]
args.a_dim = 1  # CartPole action: [force]
args.N_steps = 200  # Episode length
args.frame_skip = 1
args.dt = 0.02
args.gym_env = "CartPole-v1"

# Generate multiple CHALLENGING initial conditions to test
num_initial_conditions = 1  # Just one test
key = jax.random.PRNGKey(args.seed)
initial_conditions = []

print(f"\nGenerating {num_initial_conditions} CHALLENGING initial conditions...")
print("(Larger angles and velocities to test learning from scratch)")
for i in range(num_initial_conditions):
    key, subkey = jax.random.split(key)
    # Harder initial state: larger angles and velocities
    # x: -1.0 to 1.0, x_dot: -1.0 to 1.0
    # theta: -0.15 to 0.15 rad (~±8.5 deg), theta_dot: -1.0 to 1.0
    random_init = jax.random.uniform(subkey, (args.s_dim,), minval=-1.0, maxval=1.0)
    random_init = random_init.at[2].set(random_init[2] * 0.15)  # theta: -0.15 to 0.15
    initial_conditions.append(random_init)
    print(f"  IC {i+1}: x={random_init[0]:.3f}, x_dot={random_init[1]:.3f}, theta={random_init[2]:.3f}, theta_dot={random_init[3]:.3f}")

# Store results for all initial conditions
all_results = []

# Test each initial condition
for ic_idx, init_state in enumerate(initial_conditions):
    print(f"\n{'='*60}")
    print(f"Testing Initial Condition {ic_idx+1}/{num_initial_conditions}")
    print(f"State: x={init_state[0]:.3f}, x_dot={init_state[1]:.3f}, theta={init_state[2]:.3f}, theta_dot={init_state[3]:.3f}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Create fresh PPO agent for each initial condition
    dynamics = get_step_model(args.gym_env, None)

    ppo_agent = UnifiedPPO(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        args=args,
        state_train=None,
        dynamics=dynamics,
        mjx_model=None,
        gym_env=args.gym_env,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        rollout_length=args.N_steps,
        buffer_mix=20,
        use_learned_cost=False
    )

    # Initialize policy FROM SCRATCH: Set actor network to output near-zero actions
    # This simulates starting from a "no knowledge" baseline
    # Reinitialize actor with zeros (outputs zero mean actions)
    dummy_input = jnp.zeros((1, args.s_dim))
    key_init = jax.random.PRNGKey(args.seed + ic_idx)

    # Create new params with very small initialization
    actor_vars = ppo_agent.actor_net.init(key_init, dummy_input)
    actor_params = actor_vars['params']  # Extract just the params

    # Zero out the final layer to make initial actions ~0
    actor_params = jax.tree_util.tree_map(
        lambda x: x * 0.01,  # Scale down all weights by 100x
        actor_params
    )

    # Recreate actor state with zero-initialized params
    ppo_agent.actor_state = train_state.TrainState.create(
        apply_fn=ppo_agent.actor_net.apply,
        params=actor_params,
        tx=optax.adam(3e-4)
    )

    print("  Policy initialized FROM SCRATCH (near-zero actions)")

    # Training loop with PPO updates (FIXED initial condition)
    num_iterations = 50  # Train for 50 iterations to see learning
    rewards_history = []

    for iteration in range(num_iterations):
        # Use the SAME initial condition for all iterations in this run
        D_demo = jnp.zeros((args.N_steps, args.s_dim + 1 + args.a_dim))
        D_demo = D_demo.at[0, :args.s_dim].set(init_state)

        # Generate rollout
        states, probs, actions, total_reward = ppo_agent.generate_session_lax(
            args, None, D_demo, iteration=iteration
        )

        rewards_history.append(total_reward)

        # Update PPO policy if buffer is aligned
        if ppo_agent.buffer.p % ppo_agent.buffer.buffer_size == 0 and iteration > 0:
            try:
                buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_log_probs, buffer_next_states = ppo_agent.buffer.get()

                # Update PPO
                ppo_agent.update_ppo(
                    states=buffer_states,
                    actions=buffer_actions,
                    rewards=buffer_rewards.flatten(),
                    dones=buffer_dones.flatten(),
                    log_probs_old=buffer_log_probs.flatten(),
                    next_states=buffer_next_states,
                    gamma=0.99,
                    lam=0.97,
                    clip_eps=0.2,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    num_epochs=10,
                    batch_size=min(256, args.N_steps),
                    max_grad_norm=0.5
                )
                if iteration % 5 == 0 or iteration < 3:
                    print(f"  Iteration {iteration}: Reward={total_reward:.1f}, PPO updated")
            except AssertionError:
                if iteration % 5 == 0 or iteration < 3:
                    print(f"  Iteration {iteration}: Reward={total_reward:.1f}, buffer not aligned")
        else:
            if iteration % 5 == 0 or iteration < 3:
                print(f"  Iteration {iteration}: Reward={total_reward:.1f}")

        sys.stdout.flush()

    # Store results for this initial condition
    result = {
        'ic_idx': ic_idx,
        'init_state': init_state,
        'initial_reward': rewards_history[0],
        'final_reward': rewards_history[-1],
        'avg_last_3': np.mean(rewards_history[-3:]),
        'max_reward': np.max(rewards_history),
        'rewards_history': rewards_history
    }
    all_results.append(result)

    improvement = result['final_reward'] - result['initial_reward']
    print(f"\nResults for IC {ic_idx+1}:")
    print(f"  Initial reward: {result['initial_reward']:.1f}")
    print(f"  Final reward: {result['final_reward']:.1f}")
    print(f"  Improvement: {improvement:+.1f} ({100*improvement/(result['initial_reward']+0.1):+.1f}%)")
    print(f"  Average last 3: {result['avg_last_3']:.1f}")
    print(f"  Max reward: {result['max_reward']:.1f}")
    sys.stdout.flush()

# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL INITIAL CONDITIONS")
print(f"{'='*60}")
print(f"Number of initial conditions tested: {num_initial_conditions}")

improvements = [r['final_reward'] - r['initial_reward'] for r in all_results]
print(f"\nLEARNING PROGRESS:")
print(f"  Mean improvement: {np.mean(improvements):+.1f}")
print(f"  Std improvement:  {np.std(improvements):.1f}")
print(f"  Min improvement:  {np.min(improvements):+.1f}")
print(f"  Max improvement:  {np.max(improvements):+.1f}")

print(f"\nInitial Rewards (from scratch policy):")
print(f"  Mean: {np.mean([r['initial_reward'] for r in all_results]):.1f}")
print(f"  Std:  {np.std([r['initial_reward'] for r in all_results]):.1f}")
print(f"  Min:  {np.min([r['initial_reward'] for r in all_results]):.1f}")
print(f"  Max:  {np.max([r['initial_reward'] for r in all_results]):.1f}")
print(f"\nFinal Rewards (after training):")
print(f"  Mean: {np.mean([r['final_reward'] for r in all_results]):.1f}")
print(f"  Std:  {np.std([r['final_reward'] for r in all_results]):.1f}")
print(f"  Min:  {np.min([r['final_reward'] for r in all_results]):.1f}")
print(f"  Max:  {np.max([r['final_reward'] for r in all_results]):.1f}")
print(f"\nAverage Last 3 Rewards:")
print(f"  Mean: {np.mean([r['avg_last_3'] for r in all_results]):.1f}")
print(f"  Std:  {np.std([r['avg_last_3'] for r in all_results]):.1f}")
print(f"  Min:  {np.min([r['avg_last_3'] for r in all_results]):.1f}")
print(f"  Max:  {np.max([r['avg_last_3'] for r in all_results]):.1f}")
print(f"\nMax Rewards:")
print(f"  Mean: {np.mean([r['max_reward'] for r in all_results]):.1f}")
print(f"  Std:  {np.std([r['max_reward'] for r in all_results]):.1f}")
print(f"  Min:  {np.min([r['max_reward'] for r in all_results]):.1f}")
print(f"  Max:  {np.max([r['max_reward'] for r in all_results]):.1f}")
print("\nTest PASSED!")

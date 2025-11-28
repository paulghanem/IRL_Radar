"""
Test Simplified Walker2d with GCL + MPPI
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from flax.training import train_state
import optax
from functools import partial

from src.control.simplified_walker import SimplifiedWalker, simplified_walker_step
from src.control.mppi_class import MPPI
from cost_jax import CostNN, apply_model, update_model
from utils.helpers import GenerateDemo

print("JAX devices:", jax.devices())
print("\n" + "="*60)
print("Testing Simplified Walker2d with GCL + MPPI")
print("="*60)

# ============================================================
# Configuration
# ============================================================
class Args:
    def __init__(self):
        self.seed = 42
        self.s_dim = 18  # Walker2d state dimension (updated to match Walker2d-v4)
        self.a_dim = 6   # Walker2d action dimension
        self.N_steps = 1000  # Steps per episode
        self.rirl_iterations = 150  # Reduced for faster testing
        self.reward_fn_updates = 5  # Fewer updates for stability
        self.hidden_dim = 128  # Larger network for better capacity
        self.lr = 1e-4  # Lower learning rate for stable updates

        # MPPI parameters
        self.horizon = 20  # Moderate horizon for planning
        self.num_traj = 1000  # More samples for better optimization
        self.lambda_ = 0.001  # Moderate temperature for exploration

        # RGCL parameters - TESTING RGCL
        self.use_rgcl = True  # ENABLED for testing
        self.P_init = 5e-2  # Higher initial uncertainty for adaptation
        self.Q_noise = 1e-4  # Higher process noise for faster adaptation
        self.use_diagonal = False  # Use full covariance for accuracy

        self.gym_env = "SimplifiedWalker2d"

args = Args()

# ============================================================
# Initialize Simplified Walker2d
# ============================================================
print("\nInitializing Simplified Walker2d environment...")
walker = SimplifiedWalker()
print(f"  State dim: {walker.state_dim}")
print(f"  Action dim: {walker.action_dim}")
print(f"  Time step: {walker.dt}")

# ============================================================
# Generate Expert Demonstrations
# ============================================================
print("\nGenerating expert demonstrations...")
print("  Loading pre-trained Walker2d expert...")

# Use pre-trained expert from main.py approach
demo_generator = GenerateDemo(env_name="Walker2d", max_frames=args.N_steps)
expert_states_np, expert_actions_np, expert_rewards_cumsum, _ = demo_generator.generate_demo(seed=args.seed)

# Convert to JAX arrays
expert_states = jnp.array(expert_states_np)
expert_actions = jnp.array(expert_actions_np)

# Verify dimensions match
print(f"  Expert states shape: {expert_states.shape}, expected: ({args.N_steps}, {args.s_dim})")
print(f"  Expert actions shape: {expert_actions.shape}, expected: ({args.N_steps}, {args.a_dim})")
assert expert_states.shape == (args.N_steps, args.s_dim), f"State dimension mismatch!"
assert expert_actions.shape == (args.N_steps, args.a_dim), f"Action dimension mismatch!"

# Calculate per-step rewards from cumulative
expert_rewards = jnp.diff(expert_rewards_cumsum, prepend=0.0)

print("  Expert demonstration collection complete!")

expert_rewards_sum = jnp.sum(expert_rewards)

print(f"  Expert total reward: {expert_rewards_sum:.2f}")
print(f"  Expert avg reward: {jnp.mean(expert_rewards):.3f}")

# Prepare demo data
D_demo = jnp.concatenate([
    expert_states,
    jnp.ones((args.N_steps, 1)),  # Probs
    expert_actions
], axis=1)

# ============================================================
# Initialize Cost Function (Neural Network)
# ============================================================
print("\nInitializing cost function...")
cost_f = CostNN(state_dims=args.s_dim, hidden_dim=args.hidden_dim)
init_rng = jax.random.key(args.seed)

variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
params = variables['params']
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
    optax.adam(learning_rate=args.lr)
)
state_train = train_state.TrainState.create(
    apply_fn=cost_f.apply,
    params=params,
    tx=tx
)

@jax.jit
def cost_function(state, state_train):
    return state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel()

print("  Cost network initialized")

# ============================================================
# Initialize RGCL (if enabled)
# ============================================================
if args.use_rgcl:
    print("\nInitializing RGCL...")
    # Flatten parameters for RGCL
    flat_params, param_tree = jax.tree_util.tree_flatten(params)
    theta = jnp.concatenate([p.flatten() for p in flat_params])
    n_params = len(theta)

    # Initialize covariance matrix
    if args.use_diagonal:
        P_theta = args.P_init * jnp.ones(n_params)  # Diagonal approximation
        print(f"  Using diagonal covariance: {n_params} parameters")
    else:
        P_theta = args.P_init * jnp.eye(n_params)  # Full covariance
        print(f"  Using full covariance: {n_params}x{n_params} matrix")

    # RGCL update function
    @partial(jax.jit, static_argnames=("use_diagonal",))
    def rgcl_update(params, grads, P_theta, use_diagonal=True):
        """
        RGCL (Recursive Guided Cost Learning) parameter update
        Uses a Kalman-filter like update with Hessian approximation
        """
        # Flatten gradients
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        grad_vec = jnp.concatenate([g.flatten() for g in flat_grads])

        if use_diagonal:
            # Diagonal approximation (much faster)
            # Prediction step
            P_pred = P_theta + args.Q_noise

            # Gauss-Newton Hessian approximation: H ≈ grad @ grad.T
            H_diag = grad_vec ** 2 + 1e-8

            # Update step (Kalman gain)
            K = P_pred / (P_pred + H_diag)

            # Parameter update
            theta_new_vec = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(params)])
            theta_new_vec = theta_new_vec - K * grad_vec

            # Covariance update
            P_new = (1 - K) * P_pred
        else:
            # Full matrix version (slower but more accurate)
            P_pred = P_theta + args.Q_noise * jnp.eye(len(grad_vec))

            # Gauss-Newton Hessian
            H = jnp.outer(grad_vec, grad_vec) + 1e-8 * jnp.eye(len(grad_vec))

            # Kalman gain
            K = P_pred @ jnp.linalg.inv(P_pred + H)

            # Parameter update
            theta_vec = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(params)])
            theta_new_vec = theta_vec - K @ grad_vec

            # Covariance update
            P_new = (jnp.eye(len(grad_vec)) - K) @ P_pred

        # Unflatten parameters
        shapes = [p.shape for p in flat_params]
        new_params_flat = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_params_flat.append(theta_new_vec[idx:idx+size].reshape(shape))
            idx += size

        new_params = jax.tree_util.tree_unflatten(param_tree, new_params_flat)

        return new_params, P_new

    print("  RGCL initialized")
else:
    print("  Using standard GCL")

# ============================================================
# Initialize MPPI Policy
# ============================================================
print("\nInitializing MPPI policy...")

u_min = jnp.array([-1.0] * args.a_dim)
u_max = jnp.array([1.0] * args.a_dim)
cov_scaler = jnp.array([0.5] * args.a_dim)

policy = MPPI(
    state_train=state_train,
    horizon=args.horizon,
    num_samples=args.num_traj,
    dim_state=args.s_dim,
    dim_control=args.a_dim,
    dynamics=simplified_walker_step,
    cost_func=jax.jit(jax.vmap(cost_function, in_axes=(0, None))),
    u_min=u_min,
    u_max=u_max,
    sigmas=cov_scaler,
    lambda_=args.lambda_,
    env=None,
    mjx_model=None,
    gym_env=args.gym_env,
    use_mujoco=False
)

print(f"  MPPI initialized: {args.num_traj} samples, horizon {args.horizon}")

# ============================================================
# Training Loop
# ============================================================
print("\n" + "="*60)
print("Starting GCL + MPPI Training")
print("="*60)

agent_rewards = []
cost_losses = []
best_reward = -float('inf')
patience = 20
no_improvement_count = 0

# ========== REPLAY BUFFER (OPTIONAL - UNCOMMENT TO ENABLE) ==========
# Accumulates all past agent samples for more stable cost learning
# D_samp_buffer = None
# max_buffer_size = 50000  # Maximum number of samples in buffer
# ====================================================================

for iteration in range(args.rirl_iterations):
    print(f"\nIteration {iteration+1}/{args.rirl_iterations}")

    # Decay exploration over time
    exploration_rate = max(0.1, 1.0 - iteration / args.rirl_iterations)

    # ========== Generate Trajectory with MPPI ==========
    initial_state = walker.reset(jax.random.PRNGKey(args.seed + iteration))

    states = jnp.zeros((args.N_steps, args.s_dim))
    actions = jnp.zeros((args.N_steps, args.a_dim))
    rewards = jnp.zeros(args.N_steps)

    state = initial_state

    print("  Generating trajectory with MPPI...")
    start_rollout = time.time()

    for step in range(args.N_steps):
        # MPPI forward pass (NOTE: simplified, not using full generate_session)
        # For speed, we'll use MPPI forward directly
        action_seq, _, policy.key, policy._previous_action_seq = policy.forward_pure(
            state=state,
            state_train=state_train,
            gail=False,
            key=policy.key,
            prev_action_seq=policy._previous_action_seq,
            frame_skip=1
        )

        action = action_seq[0, :]
        next_state = walker.step(state, action)
        reward = walker.compute_reward(state, action, next_state)

        states = states.at[step].set(state)
        actions = actions.at[step].set(action)
        rewards = rewards.at[step].set(reward)

        state = next_state

    end_rollout = time.time()
    rollout_time = end_rollout - start_rollout

    total_reward = jnp.sum(rewards)

    agent_rewards.append(float(total_reward))

    print(f"  Rollout time: {rollout_time:.2f}s")
    print(f"  Agent reward: {total_reward:.2f}")

    # ========== Update Cost Function (GCL) ==========
    print("  Updating cost function...")

    # Prepare sample data
    D_samp = jnp.concatenate([
        states,
        jnp.ones((args.N_steps, 1)),  # Probs
        actions
    ], axis=1)

    # ========== REPLAY BUFFER UPDATE (UNCOMMENT TO ENABLE) ==========
    # Add current samples to replay buffer
    # if D_samp_buffer is None:
    #     D_samp_buffer = D_samp
    # else:
    #     D_samp_buffer = jnp.concatenate([D_samp_buffer, D_samp], axis=0)
    #     # Keep only most recent samples if buffer exceeds max size
    #     if D_samp_buffer.shape[0] > max_buffer_size:
    #         D_samp_buffer = D_samp_buffer[-max_buffer_size:]
    #
    # buffer_size = D_samp_buffer.shape[0]
    # print(f"  Replay buffer size: {buffer_size}")
    # ================================================================

    losses = []
    for _ in range(args.reward_fn_updates):
        # Sample mini-batches - use smaller batches for better gradient estimates
        #batch_size = min(256, args.N_steps)
        batch_size = args.N_steps
        # ========== REPLAY BUFFER SAMPLING (UNCOMMENT TO ENABLE) ==========
        # Sample from entire replay buffer instead of just current rollout
        # buffer_size = D_samp_buffer.shape[0]
        # idx_samp = np.random.choice(buffer_size, batch_size, replace=False)
        # idx_demo = np.random.choice(args.N_steps, batch_size, replace=False)
        #
        # batch_samp = D_samp_buffer[idx_samp]
        # batch_demo = D_demo[idx_demo]
        # ===================================================================

        # Standard sampling (from current rollout only)
        idx_samp = np.random.choice(args.N_steps, batch_size, replace=False)
        idx_demo = np.random.choice(args.N_steps, batch_size, replace=False)

        batch_samp = D_samp[idx_samp]
        batch_demo = D_demo[idx_demo]

        states_samp = batch_samp[:, :args.s_dim]
        probs_samp = batch_samp[:, args.s_dim]
        actions_samp = batch_samp[:, args.s_dim+1:]

        states_demo = batch_demo[:, :args.s_dim]
        probs_demo = batch_demo[:, args.s_dim]
        actions_demo = batch_demo[:, args.s_dim+1:]

        # Compute gradients with GCL loss
        grads, loss = apply_model(
            state_train, states_samp, actions_samp,
            states_demo, actions_demo,
            probs_samp, probs_demo, UB=False
        )

        # Update parameters (RGCL or standard)
        if args.use_rgcl:
            new_params, P_theta = rgcl_update(state_train.params, grads, P_theta, args.use_diagonal)
            state_train = state_train.replace(params=new_params)
        else:
            state_train = update_model(state_train, grads)

        losses.append(float(loss))

    # Propagate updated state_train to MPPI instance
    policy.state_train = state_train

    mean_loss = np.mean(losses)
    cost_losses.append(mean_loss)
    print(f"  Cost loss: {mean_loss:.4f}")

    # Track best reward and early stopping
    if total_reward > best_reward:
        best_reward = total_reward
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    # Early stopping if no improvement
    # if no_improvement_count >= patience:
    #     print(f"\n  Early stopping at iteration {iteration+1} (no improvement for {patience} iterations)")
    #     break

    # Progress summary
    if (iteration + 1) % 10 == 0:
        recent_rewards = agent_rewards[-10:]
        print(f"\n  --- Progress at iteration {iteration+1} ---")
        print(f"  Avg reward (last 10): {np.mean(recent_rewards):.2f}")
        print(f"  Best reward so far: {best_reward:.2f}")
        print(f"  Expert reward: {expert_rewards_sum:.2f}")
        print(f"  Gap to expert: {expert_rewards_sum - best_reward:.2f}")

# ============================================================
# Final Results
# ============================================================
print("\n" + "="*60)
print("Training Complete!")
print("="*60)

print(f"\nFinal Statistics:")
print(f"  Initial agent reward: {agent_rewards[0]:.2f}")
print(f"  Final agent reward: {agent_rewards[-1]:.2f}")
print(f"  Best agent reward: {max(agent_rewards):.2f}")
print(f"  Average agent reward: {np.mean(agent_rewards):.2f}")
print(f"  Expert reward: {expert_rewards_sum:.2f}")
print(f"  Achievement: {(agent_rewards[-1]/expert_rewards_sum)*100:.1f}% of expert")

print(f"\nCost Function:")
print(f"  Valid losses: {sum(1 for l in cost_losses if not np.isnan(l))}/{len(cost_losses)}")
print(f"  NaN losses: {sum(1 for l in cost_losses if np.isnan(l))}/{len(cost_losses)}")

print(f"\nExecution:")
print(f"  Total iterations: {args.rirl_iterations}")
print(f"  Steps per iteration: {args.N_steps}")
print(f"  MPPI samples: {args.num_traj}")
print(f"  MPPI horizon: {args.horizon}")

# Save results
np.save("simplified_walker_rewards.npy", agent_rewards)
np.save("simplified_walker_losses.npy", cost_losses)
print("\nResults saved:")
print("  - simplified_walker_rewards.npy")
print("  - simplified_walker_losses.npy")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)

# Standalone script to test optimized MPPI on Walker2d
import os
import sys

# Set JAX environment variables BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Configure CUDA library paths from site-packages
for path in sys.path:
    if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
        nvidia_path = os.path.join(path, 'nvidia')
        subdirs = [d for d in os.listdir(nvidia_path) if os.path.isdir(os.path.join(nvidia_path, d))]
        lib_paths = []
        for subdir in subdirs:
            lib_path = os.path.join(nvidia_path, subdir, 'lib')
            if os.path.exists(lib_path):
                lib_paths.append(lib_path)
        if lib_paths:
            existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            new_ld_path = ':'.join(lib_paths)
            if existing_ld_path:
                new_ld_path = f"{new_ld_path}:{existing_ld_path}"
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
        break

# Now import everything else
import jax
import jax.numpy as jnp
import gymnasium as gym
import time

# Import MPPI controller
from src.control.mppi_class import MPPI

# Import optimized session generator
from src.control.mppi_class_optimized import create_optimized_generate_session

# Import utilities
from src.control.dynamics import kinematics_mujoco
from cost_jax import apply_model
from mujoco import mjx
import flax.linen as nn
from flax.training import train_state
import optax

print("=" * 60)
print("Walker2d RDIRL with Optimized MPPI (lax.scan-based)")
print("=" * 60)
print("")

# Setup experiment parameters
class Args:
    gym_env = "Walker2d-v4"
    num_traj = 500
    horizon = 50
    N_steps = 100
    lambda_ = 0.01
    frame_skip = 4
    seed = 123
    lr = 1e-4
    Q = 1e-4
    P = 1e-2
    hidden_dim = 16

args = Args()

# Create environment
print("Creating Walker2d-v4 environment...")
env = gym.make(args.gym_env)
args.s_dim = env.observation_space.shape[0]
args.a_dim = env.action_space.shape[0]
args.dt = env.dt

# Get action space bounds
u_min = jnp.array(env.action_space.low)
u_max = jnp.array(env.action_space.high)
cov_scaler = (u_max - u_min) ** 2 / 16  # 1D vector, not matrix

print(f"State dim: {args.s_dim}, Action dim: {args.a_dim}")
print(f"N_steps: {args.N_steps}, Horizon: {args.horizon}, Num trajectories: {args.num_traj}")
print("")

# Create MuJoCo model for dynamics
print("Loading MuJoCo model...")
mjx_model = mjx.put_model(env.unwrapped.model)
mjx_data = mjx.put_data(env.unwrapped.model, env.unwrapped.data)

# Create cost network
print("Initializing cost network...")
class CostNetwork(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

cost_f = CostNetwork(hidden_dim=args.hidden_dim)
init_rng = jax.random.PRNGKey(args.seed)
variables = cost_f.init(init_rng, jnp.ones((1, args.s_dim)))
params = variables['params']
tx = optax.adam(learning_rate=args.lr)
state_train = train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)

# Define cost function
def cost_function(state, state_train):
    """Compute cost for a state using the learned cost network"""
    return apply_model(state_train, state)[0]

# Create MPPI controller
print("Creating MPPI controller...")
policy = MPPI(
    state_train=state_train,
    horizon=args.horizon,
    num_samples=args.num_traj,
    dim_state=args.s_dim,
    dim_control=args.a_dim,
    dynamics=kinematics_mujoco,
    cost_func=jax.jit(jax.vmap(cost_function, in_axes=(0, None))),
    u_min=u_min,
    u_max=u_max,
    sigmas=cov_scaler,
    lambda_=args.lambda_,
    env=env,
    mjx_model=mjx_model,
    gym_env=args.gym_env,
    use_mujoco=True
)
print("MPPI controller created")
print("")

# Create optimized session generator
print("Creating optimized lax.scan-based session generator...")
generate_session_optimized = create_optimized_generate_session(policy)
print("Optimized generator created")
print("")

# Initialize environment
print("Initializing environment...")
key = jax.random.PRNGKey(args.seed)
state, info = env.reset(seed=args.seed)
state = jnp.array(state, dtype=jnp.float32)
prev_action_seq = jnp.zeros((args.horizon, args.a_dim))

print("=" * 60)
print("Running optimized session generation (lax.scan N_steps loop)...")
print("=" * 60)
print("")

# Time the execution
start_time = time.time()

states, actions, total_reward, final_prev_action_seq = generate_session_optimized(
    state,
    key,
    prev_action_seq,
    state_train,
    args.N_steps,
    args.gym_env,
    args.frame_skip,
    args.dt,
    gail=False
)

# Block until computation completes
states.block_until_ready()

end_time = time.time()
execution_time = end_time - start_time

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"N_steps lax.scan outer loop execution time: {execution_time:.4f} seconds")
print(f"Total reward: {float(total_reward):.4f}")
print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print("")
print("Experiment complete")

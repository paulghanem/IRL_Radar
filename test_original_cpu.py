# Test original MPPI on CPU to compare with optimized version
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import gymnasium as gym
from src.control.mppi_class import MPPI
from src.control.dynamics import kinematics_mujoco
from cost_jax import apply_model
from mujoco import mjx
import flax.linen as nn
from flax.training import train_state
import optax
import time

print("=" * 60)
print("CPU Test: Original MPPI")
print("=" * 60)
print("")

# Same minimal parameters as optimized test
class Args:
    gym_env = "Walker2d-v4"
    num_traj = 10
    horizon = 5
    N_steps = 3
    lambda_ = 0.01
    frame_skip = 4
    seed = 123
    lr = 1e-4
    Q = 1e-4
    P = 1e-2
    hidden_dim = 8

args = Args()

print("Creating environment...")
env = gym.make(args.gym_env)
args.s_dim = env.observation_space.shape[0]
args.a_dim = env.action_space.shape[0]
args.dt = env.dt

u_min = jnp.array(env.action_space.low)
u_max = jnp.array(env.action_space.high)
cov_scaler = (u_max - u_min) ** 2 / 16

print(f"State dim: {args.s_dim}, Action dim: {args.a_dim}")
print(f"N_steps: {args.N_steps}, Horizon: {args.horizon}, Num trajectories: {args.num_traj}")
print("")

print("Loading MuJoCo model...")
mjx_model = mjx.put_model(env.unwrapped.model)
mjx_data = mjx.put_data(env.unwrapped.model, env.unwrapped.data)

print("Creating cost network...")
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

def cost_function(state, state_train):
    return apply_model(state_train, state)[0]

print("Creating original MPPI controller...")
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
print("✓ MPPI controller created")
print("")

print("Initializing environment...")
key = jax.random.PRNGKey(args.seed)
state, info = env.reset(seed=args.seed)
state = jnp.array(state, dtype=jnp.float32)

print(f"Running original MPPI test with N_steps={args.N_steps}...")
print("")

try:
    start_time = time.time()

    states = []
    actions_list = []
    total_reward = 0.0

    for t in range(args.N_steps):
        # Use original MPPI forward method
        action = policy.forward(state, key, args.gym_env, args.frame_skip, gail=False)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = jnp.array(next_state, dtype=jnp.float32)

        states.append(state)
        actions_list.append(action)
        total_reward += reward

        state = next_state
        if terminated or truncated:
            break

    end_time = time.time()
    execution_time = end_time - start_time

    states = jnp.array(states)
    actions = jnp.array(actions_list)

    print("=" * 60)
    print("✓ ORIGINAL MPPI TEST SUCCESSFUL!")
    print("=" * 60)
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Total reward: {float(total_reward):.4f}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print("")
    print("Original MPPI works on CPU. The issue must be in the optimized version.")
    print("")

except Exception as e:
    print("=" * 60)
    print("✗ ORIGINAL MPPI TEST ALSO FAILED")
    print("=" * 60)
    print(f"Error: {e}")
    print("")
    import traceback
    traceback.print_exc()
    print("")
    print("This suggests the issue is not specific to the optimized version.")

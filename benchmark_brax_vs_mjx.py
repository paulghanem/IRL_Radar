"""
Benchmark script comparing Brax vs MJX performance for MPPI rollouts
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import mujoco
from mujoco import mjx
from brax import envs
from src.control.mppi_class import MPPI
from src.control.dynamics import get_step_model
from cost_jax import CostNN
import flax
from flax.training import train_state
import optax

print("JAX devices:", jax.devices())
print("=" * 80)

# Configuration
ENV_NAME = "HalfCheetah-v4"
HORIZON = 50
NUM_SAMPLES = 2000
NUM_ITERATIONS = 10

print(f"Benchmark Configuration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Horizon: {HORIZON}")
print(f"  Num Samples: {NUM_SAMPLES}")
print(f"  Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# Environment setup
if ENV_NAME == "HalfCheetah-v4":
    env_xml = "assets/half_cheetah.xml"
    frame_skip = 5
    dt = 0.01
    brax_env_name = 'halfcheetah'
    s_dim = 17
    a_dim = 6
elif ENV_NAME == "Ant":
    env_xml = "assets/ant.xml"
    frame_skip = 5
    dt = 0.01
    brax_env_name = 'ant'
    s_dim = 27
    a_dim = 8
elif ENV_NAME == "Hopper":
    env_xml = "assets/hopper.xml"
    frame_skip = 4
    dt = 0.002
    brax_env_name = 'hopper'
    s_dim = 11
    a_dim = 3
else:
    raise ValueError(f"Unknown environment: {ENV_NAME}")

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(env_xml)
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

# Load Brax environment
env_brax = envs.get_environment(brax_env_name)

# Create dummy cost network
hidden_dim = 16
cost_nn = CostNN(state_dims=s_dim, hidden_dim=hidden_dim)
rng = jax.random.PRNGKey(0)
dummy_state = jnp.ones((1, s_dim))
params = cost_nn.init(rng, dummy_state)

# Cost function
def cost_function(state, state_train):
    output = cost_nn.apply(state_train.params, state)
    return output

# Training state
tx = optax.adam(learning_rate=1e-4)
state_train = train_state.TrainState.create(
    apply_fn=cost_nn.apply,
    params=params,
    tx=tx
)

# Action space (hardcoded for benchmark)
if ENV_NAME == "HalfCheetah-v4":
    u_min = jnp.array([-1.0] * a_dim)
    u_max = jnp.array([1.0] * a_dim)
    cov_scaler = jnp.array([0.5] * a_dim)
elif ENV_NAME == "Ant":
    u_min = jnp.array([-1.0] * a_dim)
    u_max = jnp.array([1.0] * a_dim)
    cov_scaler = jnp.array([0.5] * a_dim)
elif ENV_NAME == "Hopper":
    u_min = jnp.array([-1.0] * a_dim)
    u_max = jnp.array([1.0] * a_dim)
    cov_scaler = jnp.array([0.5] * a_dim)
else:
    u_min = jnp.array([-1.0] * a_dim)
    u_max = jnp.array([1.0] * a_dim)
    cov_scaler = jnp.array([0.5] * a_dim)

# Create MPPI with MJX (env_brax=None)
print("Creating MPPI with MJX...")
mppi_mjx = MPPI(
    state_train=state_train,
    horizon=HORIZON,
    num_samples=NUM_SAMPLES,
    dim_state=s_dim,
    dim_control=a_dim,
    dynamics=get_step_model(ENV_NAME, None),
    cost_func=jax.jit(jax.vmap(cost_function, in_axes=(0, None))),
    u_min=u_min,
    u_max=u_max,
    sigmas=cov_scaler,
    lambda_=0.01,
    env=None,
    mjx_model=mjx_model,
    gym_env=ENV_NAME,
    env_brax=None,  # Disable Brax
    use_mujoco=True
)

# Create MPPI with Brax
print("Creating MPPI with Brax...")
mppi_brax = MPPI(
    state_train=state_train,
    horizon=HORIZON,
    num_samples=NUM_SAMPLES,
    dim_state=s_dim,
    dim_control=a_dim,
    dynamics=get_step_model(ENV_NAME, None),
    cost_func=jax.jit(jax.vmap(cost_function, in_axes=(0, None))),
    u_min=u_min,
    u_max=u_max,
    sigmas=cov_scaler,
    lambda_=0.01,
    env=None,
    mjx_model=mjx_model,
    gym_env=ENV_NAME,
    env_brax=env_brax,  # Enable Brax
    use_mujoco=True
)

# Initialize test state
test_state = jnp.ones(s_dim)
key = jax.random.PRNGKey(42)
prev_action_seq = jnp.zeros((HORIZON, a_dim))

print("=" * 80)
print("Warming up JIT compilation...")

# Warm-up for MJX
key, subkey = jax.random.split(key)
_ = mppi_mjx.forward_pure(
    state=test_state,
    state_train=state_train,
    gail=False,
    key=subkey,
    prev_action_seq=prev_action_seq,
    frame_skip=frame_skip
)
print("  MJX warm-up complete")

# Warm-up for Brax
key, subkey = jax.random.split(key)
brax_state = env_brax.reset(subkey)
brax_state = brax_state.replace(obs=test_state)
_ = mppi_brax.forward_pure_brax(
    state=test_state,
    state_train=state_train,
    gail=False,
    key=subkey,
    prev_action_seq=prev_action_seq,
    frame_skip=frame_skip,
    brax_state0=brax_state
)
print("  Brax warm-up complete")

print("=" * 80)
print(f"Running {NUM_ITERATIONS} iterations...")
print()

# Benchmark MJX
print("Benchmarking MJX...")
mjx_times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    start = time.time()
    _, _, _, _ = mppi_mjx.forward_pure(
        state=test_state,
        state_train=state_train,
        gail=False,
        key=subkey,
        prev_action_seq=prev_action_seq,
        frame_skip=frame_skip
    )
    # Block until computation is complete
    jax.block_until_ready(_)
    end = time.time()
    mjx_times.append(end - start)
    print(f"  Iteration {i+1}/{NUM_ITERATIONS}: {mjx_times[-1]:.4f}s")

# Benchmark Brax
print("\nBenchmarking Brax...")
brax_times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    brax_state = env_brax.reset(subkey)
    brax_state = brax_state.replace(obs=test_state)

    start = time.time()
    _, _, _, _ = mppi_brax.forward_pure_brax(
        state=test_state,
        state_train=state_train,
        gail=False,
        key=subkey,
        prev_action_seq=prev_action_seq,
        frame_skip=frame_skip,
        brax_state0=brax_state
    )
    # Block until computation is complete
    jax.block_until_ready(_)
    end = time.time()
    brax_times.append(end - start)
    print(f"  Iteration {i+1}/{NUM_ITERATIONS}: {brax_times[-1]:.4f}s")

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

mjx_mean = np.mean(mjx_times)
mjx_std = np.std(mjx_times)
brax_mean = np.mean(brax_times)
brax_std = np.std(brax_times)

print(f"\nMJX Performance:")
print(f"  Mean: {mjx_mean:.4f}s ± {mjx_std:.4f}s")
print(f"  Min:  {np.min(mjx_times):.4f}s")
print(f"  Max:  {np.max(mjx_times):.4f}s")

print(f"\nBrax Performance:")
print(f"  Mean: {brax_mean:.4f}s ± {brax_std:.4f}s")
print(f"  Min:  {np.min(brax_times):.4f}s")
print(f"  Max:  {np.max(brax_times):.4f}s")

speedup = mjx_mean / brax_mean
print(f"\nSpeedup: {speedup:.2f}x")

if speedup > 1.0:
    print(f"✓ Brax is {speedup:.2f}x FASTER than MJX")
else:
    print(f"✗ Brax is {1/speedup:.2f}x SLOWER than MJX")

print("\n" + "=" * 80)
print("Benchmark complete!")

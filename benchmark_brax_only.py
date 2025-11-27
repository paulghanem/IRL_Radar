"""
Benchmark script for Brax-only performance on GPU
This tests if Brax works properly on GPU without MJX complications
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
from brax import envs

print("=" * 80)
print("BRAX-ONLY GPU BENCHMARK")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")
print("=" * 80)

# Configuration
ENV_NAME = "HalfCheetah-v4"
BRAX_ENV_NAME = 'halfcheetah'
HORIZON = 50
NUM_SAMPLES = 2000
NUM_ITERATIONS = 10

print(f"\nBenchmark Configuration:")
print(f"  Environment: {ENV_NAME} (Brax: {BRAX_ENV_NAME})")
print(f"  Horizon: {HORIZON}")
print(f"  Num Samples: {NUM_SAMPLES}")
print(f"  Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# Load Brax environment
print("\nLoading Brax environment...")
env_brax = envs.get_environment(BRAX_ENV_NAME)
key = jax.random.PRNGKey(42)
state = env_brax.reset(key)
action_dim = env_brax.action_size
obs_dim = state.obs.shape[0]

print(f"  Action dim: {action_dim}")
print(f"  Observation dim: {obs_dim}")

# Define Brax rollout function
@jax.jit
def rollout_brax(env, init_state, action_seqs):
    """
    Brax-style rollout: vmap over samples, scan over horizon
    Args:
        env: Brax environment
        init_state: Initial environment state
        action_seqs: (num_samples, horizon, action_dim)
    Returns:
        obs_batch: (num_samples, horizon+1, obs_dim)
    """
    def rollout_single(actions_seq):
        def step_fn(state, action):
            next_state = env.step(state, action)
            return next_state, next_state.obs

        final_state, obs_seq = jax.lax.scan(
            step_fn,
            init_state,
            actions_seq
        )
        # Prepend initial observation
        obs_with_init = jnp.concatenate([init_state.obs[None, :], obs_seq], axis=0)
        return obs_with_init

    # Vmap over batch dimension
    obs_batch = jax.vmap(rollout_single)(action_seqs)
    return obs_batch

print("\n" + "=" * 80)
print("WARMING UP (JIT compilation)")
print("=" * 80)

# Generate random action sequences for warm-up
key, subkey = jax.random.split(key)
actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
actions = jnp.clip(actions, -1.0, 1.0)

print("  Compiling Brax rollout...")
_ = rollout_brax(env_brax, state, actions)
print("  ✓ Compilation complete")

print("\n" + "=" * 80)
print(f"RUNNING {NUM_ITERATIONS} BENCHMARK ITERATIONS")
print("=" * 80)

times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
    actions = jnp.clip(actions, -1.0, 1.0)

    start = time.time()
    result = rollout_brax(env_brax, state, actions)
    result.block_until_ready()  # Wait for GPU computation
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Iteration {i+1}/{NUM_ITERATIONS}: {elapsed:.4f}s")

# Calculate statistics
mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)
throughput = (NUM_SAMPLES * HORIZON) / mean_time

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nBrax Performance on GPU:")
print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
print(f"  Min:  {min_time:.4f}s")
print(f"  Max:  {max_time:.4f}s")
print(f"  Throughput: {throughput:.0f} steps/second")
print(f"  Total steps per iteration: {NUM_SAMPLES * HORIZON:,}")

print("\n" + "=" * 80)
if jax.default_backend() == 'gpu':
    print("✓ Successfully running on GPU!")
else:
    print("⚠️  Running on", jax.default_backend())
print("=" * 80)

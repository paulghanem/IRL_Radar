"""
Fixed Benchmark: Brax vs MJX Performance Comparison on GPU
Properly handles state dimensions for both implementations
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import mujoco
from mujoco import mjx
from brax import envs

print("=" * 80)
print("BRAX vs MJX GPU PERFORMANCE BENCHMARK (Fixed)")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print("=" * 80)

# Configuration
ENV_NAME = "HalfCheetah-v4"
BRAX_ENV_NAME = 'halfcheetah'
HORIZON = 50
NUM_SAMPLES = 2000
NUM_ITERATIONS = 10

print(f"\nConfiguration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Horizon: {HORIZON}")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Iterations: {NUM_ITERATIONS}")
print("=" * 80)

# Load environments
print(f"\nLoading MuJoCo/MJX model...")
model = mujoco.MjModel.from_xml_path("assets/half_cheetah.xml")
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)

print(f"  MJX dimensions:")
print(f"    nq (qpos): {mjx_model.nq}")
print(f"    nv (qvel): {mjx_model.nv}")
print(f"    State dim: {mjx_model.nq + mjx_model.nv}")
print(f"  ✓ MJX model loaded")

print(f"\nLoading Brax environment...")
env_brax = envs.get_environment(BRAX_ENV_NAME)
key = jax.random.PRNGKey(42)
brax_state = env_brax.reset(key)
action_dim = env_brax.action_size
brax_obs_dim = brax_state.obs.shape[0]

print(f"  Action dim: {action_dim}")
print(f"  Brax obs dim: {brax_obs_dim}")
print(f"  ✓ Brax environment loaded")

# Create initial states
# For HalfCheetah: Brax obs is 17D, MJX needs 18D (9 qpos + 9 qvel)
# Brax excludes root x position, MJX includes it
print(f"\nInitializing states...")
mjx_initial_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])  # 18D for MJX
print(f"  MJX initial state shape: {mjx_initial_state.shape}")

# ========================================
# MJX Rollout Function
# ========================================
from functools import partial

@partial(jax.jit, static_argnames=('frame_skip',))
def rollout_mjx(mjx_model, init_mjx_data, init_state, action_seqs, frame_skip=5):
    """
    MJX rollout using vmap and scan
    Args:
        mjx_model: MJX model
        init_mjx_data: Initial MJX data
        init_state: (num_samples, 18) - full MJX state [qpos, qvel]
        action_seqs: (num_samples, horizon, action_dim)
        frame_skip: physics substeps per action
    Returns:
        states: (num_samples, horizon+1, 18)
    """
    def rollout_single(init_state_single, actions_seq):
        # Initialize mjx_data from state
        mjx_data = init_mjx_data.replace(
            qpos=init_state_single[:mjx_model.nq],
            qvel=init_state_single[mjx_model.nq:]
        )

        def one_step(carry_data, action):
            # Substep function for frame_skip
            def substep(d, _):
                d = d.replace(ctrl=action)
                d = mjx.step(mjx_model, d)
                return d, None

            # Run frame_skip physics steps
            carry_data, _ = jax.lax.scan(substep, carry_data, xs=None, length=frame_skip)

            # Return state as [qpos, qvel]
            state = jnp.concatenate([carry_data.qpos, carry_data.qvel])
            return carry_data, state

        # Scan over horizon
        final_data, states = jax.lax.scan(one_step, mjx_data, actions_seq)

        # Prepend initial state
        states_with_init = jnp.concatenate([init_state_single[None, :], states], axis=0)
        return states_with_init

    # Vmap over batch dimension
    states_batch = jax.vmap(rollout_single, in_axes=(0, 0))(init_state, action_seqs)
    return states_batch

# ========================================
# Brax Rollout Function
# ========================================
def make_rollout_fn_brax(env):
    @jax.jit
    def rollout_brax(init_state, action_seqs):
        """
        Brax rollout using vmap and scan
        Args:
            init_state: Brax state
            action_seqs: (num_samples, horizon, action_dim)
        Returns:
            obs_batch: (num_samples, horizon+1, obs_dim)
        """
        def rollout_single(actions_seq):
            def step_fn(state, action):
                next_state = env.step(state, action)
                return next_state, next_state.obs

            final_state, obs_seq = jax.lax.scan(step_fn, init_state, actions_seq)
            obs_with_init = jnp.concatenate([init_state.obs[None, :], obs_seq], axis=0)
            return obs_with_init

        obs_batch = jax.vmap(rollout_single)(action_seqs)
        return obs_batch

    return rollout_brax

rollout_brax = make_rollout_fn_brax(env_brax)

# ========================================
# Benchmark MJX
# ========================================
print("\n" + "=" * 80)
print("MJX BENCHMARK")
print("=" * 80)

# Generate random actions
key, subkey = jax.random.split(key)
actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
actions = jnp.clip(actions, -1.0, 1.0)

# Prepare initial states (tiled for all samples)
init_states_mjx = jnp.tile(mjx_initial_state, (NUM_SAMPLES, 1))

# Warm-up
print("  Compiling MJX rollout...")
_ = rollout_mjx(mjx_model, mjx_data, init_states_mjx, actions, frame_skip=5)
print("  ✓ Compilation complete")

# Benchmark
print(f"  Running {NUM_ITERATIONS} iterations...")
mjx_times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
    actions = jnp.clip(actions, -1.0, 1.0)

    start = time.time()
    states = rollout_mjx(mjx_model, mjx_data, init_states_mjx, actions, frame_skip=5)
    states.block_until_ready()
    elapsed = time.time() - start
    mjx_times.append(elapsed)
    print(f"    Iteration {i+1}/{NUM_ITERATIONS}: {elapsed:.4f}s")

# ========================================
# Benchmark Brax
# ========================================
print("\n" + "=" * 80)
print("BRAX BENCHMARK")
print("=" * 80)

# Warm-up
key, subkey = jax.random.split(key)
actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
actions = jnp.clip(actions, -1.0, 1.0)

print("  Compiling Brax rollout...")
_ = rollout_brax(brax_state, actions)
print("  ✓ Compilation complete")

# Benchmark
print(f"  Running {NUM_ITERATIONS} iterations...")
brax_times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    actions = jax.random.normal(subkey, (NUM_SAMPLES, HORIZON, action_dim))
    actions = jnp.clip(actions, -1.0, 1.0)

    start = time.time()
    obs = rollout_brax(brax_state, actions)
    obs.block_until_ready()
    elapsed = time.time() - start
    brax_times.append(elapsed)
    print(f"    Iteration {i+1}/{NUM_ITERATIONS}: {elapsed:.4f}s")

# ========================================
# Results
# ========================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

mjx_mean = np.mean(mjx_times)
mjx_std = np.std(mjx_times)
mjx_min = np.min(mjx_times)
mjx_max = np.max(mjx_times)
mjx_throughput = (NUM_SAMPLES * HORIZON) / mjx_mean

brax_mean = np.mean(brax_times)
brax_std = np.std(brax_times)
brax_min = np.min(brax_times)
brax_max = np.max(brax_times)
brax_throughput = (NUM_SAMPLES * HORIZON) / brax_mean

print(f"\nMJX Performance on GPU:")
print(f"  Mean time:    {mjx_mean:.4f}s ± {mjx_std:.4f}s")
print(f"  Min time:     {mjx_min:.4f}s")
print(f"  Max time:     {mjx_max:.4f}s")
print(f"  Throughput:   {mjx_throughput:,.0f} steps/second")

print(f"\nBrax Performance on GPU:")
print(f"  Mean time:    {brax_mean:.4f}s ± {brax_std:.4f}s")
print(f"  Min time:     {brax_min:.4f}s")
print(f"  Max time:     {brax_max:.4f}s")
print(f"  Throughput:   {brax_throughput:,.0f} steps/second")

# Comparison
speedup = mjx_mean / brax_mean
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if speedup > 1.0:
    print(f"\n✓ Brax is {speedup:.2f}x FASTER than MJX")
    print(f"  MJX:  {mjx_mean:.4f}s per iteration")
    print(f"  Brax: {brax_mean:.4f}s per iteration")
else:
    print(f"\n✓ MJX is {1/speedup:.2f}x FASTER than Brax")
    print(f"  MJX:  {mjx_mean:.4f}s per iteration")
    print(f"  Brax: {brax_mean:.4f}s per iteration")

print(f"\nPractical implications (1000 training iterations):")
print(f"  MJX:  {mjx_mean * 1000:.1f}s ({mjx_mean * 1000 / 60:.1f} minutes)")
print(f"  Brax: {brax_mean * 1000:.1f}s ({brax_mean * 1000 / 60:.1f} minutes)")
print(f"  Time saved: {abs(mjx_mean - brax_mean) * 1000 / 60:.1f} minutes")

print("\n" + "=" * 80)
print("✓ Both MJX and Brax are GPU-accelerated and working!")
print("=" * 80)

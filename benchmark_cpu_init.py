"""
Benchmark with CPU initialization workaround for cuDNN issues
"""
import os
# Force CPU for initialization to avoid cuDNN issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import time
import mujoco
from mujoco import mjx
from brax import envs

print("=" * 80)
print("INITIALIZING WITH CPU (cuDNN workaround)")
print("=" * 80)
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Configuration
ENV_NAME = "HalfCheetah-v4"
BRAX_ENV_NAME = 'halfcheetah'
HORIZON = 50
NUM_SAMPLES = 2000
NUM_ITERATIONS = 10

print(f"\nLoading models on CPU...")

# Load MuJoCo model on CPU
model = mujoco.MjModel.from_xml_path("assets/half_cheetah.xml")
print(f"  ✓ MuJoCo model loaded")

# Load MJX model on CPU
mjx_model = mjx.put_model(model)
mjx_data = mjx.make_data(mjx_model)
print(f"  ✓ MJX model loaded")

# Load Brax environment on CPU
env_brax = envs.get_environment(BRAX_ENV_NAME)
print(f"  ✓ Brax environment loaded")

# Now switch to GPU for actual computation
print(f"\nSwitching to GPU for computation...")
os.environ['JAX_PLATFORMS'] = 'gpu'

# Force JAX to reinitialize (this is tricky, but we'll see if it works)
import importlib
import jax._src.xla_bridge as xb
try:
    # Clear cached backend
    xb._backends = {}
    xb._backend_errors = {}
    xb._default_backend = None
    print("  Cleared JAX backend cache")
except Exception as e:
    print(f"  Warning: Could not clear JAX cache: {e}")

# Test if we can now use GPU
print(f"  Current devices after switch: {jax.devices()}")
print(f"  Current backend: {jax.default_backend()}")

if jax.default_backend() == 'cpu':
    print("\n⚠️  Still on CPU - GPU switch didn't work")
    print("   Running benchmark on CPU instead")
else:
    print("\n✓ Successfully switched to GPU!")

print("\n" + "=" * 80)
print("BENCHMARK")
print("=" * 80)

# Test simple Brax rollout
key = jax.random.PRNGKey(42)
state = env_brax.reset(key)
action_dim = env_brax.action_size

@jax.jit
def simple_rollout(init_state, actions):
    """Single trajectory rollout"""
    def step_fn(state, action):
        next_state = env_brax.step(state, action)
        return next_state, next_state.obs

    final_state, obs_seq = jax.lax.scan(step_fn, init_state, actions)
    return obs_seq

# Warm-up
key, subkey = jax.random.split(key)
actions = jax.random.normal(subkey, (HORIZON, action_dim))
actions = jnp.clip(actions, -1.0, 1.0)

print("  Compiling...")
_ = simple_rollout(state, actions)
print("  ✓ Compiled")

# Benchmark
print(f"  Running {NUM_ITERATIONS} iterations...")
times = []
for i in range(NUM_ITERATIONS):
    key, subkey = jax.random.split(key)
    actions = jax.random.normal(subkey, (HORIZON, action_dim))
    actions = jnp.clip(actions, -1.0, 1.0)

    start = time.time()
    result = simple_rollout(state, actions)
    result.block_until_ready()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"    Iteration {i+1}: {elapsed:.4f}s")

mean_time = np.mean(times)
print(f"\n  Mean time: {mean_time:.4f}s")
print(f"  Backend: {jax.default_backend()}")

print("\n" + "=" * 80)
print("Note: This approach loaded models on CPU to avoid cuDNN issues.")
print("The cuDNN version mismatch (9.1.0 vs 9.8.0) prevents GPU initialization.")
print("To fix this properly, you would need to:")
print("  1. Update jaxlib to match cuDNN 9.1.0, or")
print("  2. Update system cuDNN to 9.8.0, or")
print("  3. Use a compatible CUDA module")
print("=" * 80)

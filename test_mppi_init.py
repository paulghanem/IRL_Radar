"""
Test MPPI initialization with simplified HalfCheetah
"""

print("Step 1: Importing JAX...")
import jax
import jax.numpy as jnp

print("Step 2: Importing simplified HalfCheetah...")
from src.control.simplified_halfcheetah import simplified_halfcheetah_step

print("Step 3: Importing cost function...")
from flax.training import train_state
import optax
from cost_jax import CostNN

print("Step 4: Initializing cost network...")
s_dim = 17
a_dim = 6
cost_f = CostNN(state_dims=s_dim, hidden_dim=64)
init_rng = jax.random.key(42)
variables = cost_f.init(init_rng, jnp.ones((1, s_dim)))
params = variables['params']
tx = optax.adam(learning_rate=1e-4)
state_train = train_state.TrainState.create(
    apply_fn=cost_f.apply,
    params=params,
    tx=tx
)

@jax.jit
def cost_function(state, state_train):
    return state_train.apply_fn({'params': state_train.params}, state.reshape(1, -1)).ravel()

print("Step 5: Importing MPPI class...")
from src.control.mppi_class import MPPI

print("Step 6: Preparing MPPI parameters...")
u_min = jnp.array([-1.0] * a_dim)
u_max = jnp.array([1.0] * a_dim)
cov_scaler = jnp.array([0.5] * a_dim)

print("Step 7: Initializing MPPI...")
policy = MPPI(
    state_train=state_train,
    horizon=50,
    num_samples=500,
    dim_state=s_dim,
    dim_control=a_dim,
    dynamics=simplified_halfcheetah_step,
    cost_func=jax.jit(jax.vmap(cost_function, in_axes=(0, None))),
    u_min=u_min,
    u_max=u_max,
    sigmas=cov_scaler,
    lambda_=0.01,
    env=None,
    mjx_model=None,
    gym_env="SimplifiedHalfCheetah",
    use_mujoco=False
)

print("Step 8: MPPI initialized successfully!")
print("Test completed!")

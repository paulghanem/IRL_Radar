# How to Improve Convergence for GCL + MPPI Training

## Changes Already Applied ✅

### 1. MPPI Parameters
- **Horizon**: 50 → 100 (better long-term planning)
- **Num Samples**: 500 → 1000 (more accurate optimization)
- **Lambda**: 0.01 → 0.001 (sharper action distribution)

### 2. Cost Network
- **Hidden Dim**: 16 → 128 (more capacity to learn complex costs)
- **Learning Rate**: 1e-4 → 3e-4 (faster learning)
- **Gradient Clipping**: Added (prevents exploding gradients)

### 3. Training
- **Iterations**: 100 → 200 (more training time)
- **Cost Updates**: 15 → 30 per iteration (better cost learning)
- **Batch Size**: 1000 → 256 (better gradient estimates)
- **Early Stopping**: Added (stops if no improvement for 20 iterations)

---

## Additional Tips to Try

### If Still Not Converging:

#### A. **Adjust MPPI Temperature (lambda_)**
```python
self.lambda_ = 0.01   # Current: 0.001
# Higher = more exploration, Lower = more exploitation
# Try: 0.005, 0.01, 0.05 depending on performance
```

#### B. **Increase MPPI Samples Further**
```python
self.num_traj = 2000  # Current: 1000
# More samples = better but slower
# GPU can handle 5000+
```

#### C. **Use Learning Rate Schedule**
```python
# In optimizer setup:
schedule = optax.exponential_decay(
    init_value=3e-4,
    transition_steps=1000,
    decay_rate=0.95
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule)
)
```

#### D. **Warm Start with True Reward**
```python
# First 10 iterations: use true reward to initialize cost
if iteration < 10:
    # Train cost to match true walker reward
    true_rewards = [walker.compute_reward(s, a, ns)
                    for s, a, ns in zip(states, actions, next_states)]
```

#### E. **Better Simplified Dynamics**
The simplified Walker2d is a rough approximation. Consider:
- Using actual MuJoCo Walker2d (slower but accurate)
- Improving simplified dynamics with:
  - Better physics modeling
  - Learning a dynamics model
  - Using MJX (MuJoCo in JAX) for speed + accuracy

#### F. **Cost Function Architecture**
Try different architectures:
```python
# Current: 2-layer MLP
# Try: Deeper network
class CostNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.clip(x**2, 0, 5)
```

#### G. **Use RGCL (Recursive Guided Cost Learning)** ✨ NEW
RGCL uses a Kalman-filter-like update with Hessian approximation for more stable convergence:

```python
# In Args class, set:
self.use_rgcl = True  # Enable RGCL
self.P_init = 1e-2    # Initial covariance
self.Q_noise = 1e-5   # Process noise
self.use_diagonal = True  # Fast diagonal approximation
```

**RGCL Benefits:**
- ✅ More stable than standard GCL
- ✅ Adaptive learning rates (like second-order methods)
- ✅ Better handling of curvature
- ✅ Faster convergence in many cases

**RGCL Tuning:**
- `P_init`: Higher = more initial uncertainty → more exploration (try 1e-2 to 1e-1)
- `Q_noise`: Higher = faster adaptation but less stable (try 1e-6 to 1e-4)
- `use_diagonal`: True for speed, False for accuracy

**When to use RGCL:**
- Standard GCL is unstable or oscillating
- Need faster convergence
- Have enough compute (slightly slower than GCL)

#### H. **Use GAIL Instead of GCL**
GCL can be unstable. Try GAIL (set `gail=True` in forward_pure):
```python
action_seq, _, policy.key, policy._previous_action_seq = policy.forward_pure(
    state=state,
    state_train=state_train,
    gail=True,  # Change this
    key=policy.key,
    prev_action_seq=policy._previous_action_seq,
    frame_skip=1
)
```

---

## Debugging Checklist

### Check These if Training Fails:

1. **Cost Loss**
   - Should decrease over time
   - If NaN: reduce learning rate or add gradient clipping
   - If stuck: increase learning rate or network size

2. **Agent Rewards**
   - Should trend upward (with variance)
   - If decreasing: cost function might be learning wrong objective
   - If flat: need more exploration or better MPPI params

3. **State Mismatch**
   - Ensure expert demos match simplified dynamics
   - Simplified Walker2d ≠ Real Walker2d physics
   - Consider using real MuJoCo for expert + agent

4. **MPPI Performance**
   - Check rollout time (should be <30s)
   - If too slow: reduce horizon or num_samples
   - If actions seem random: increase lambda_ (more exploration)

---

## Expected Performance

### Realistic Targets:

- **Early iterations (1-20)**: 100-500 reward
- **Mid training (20-100)**: 500-800 reward
- **Late training (100-200)**: 800-1000 reward
- **Expert level**: 1108 reward

### Convergence Indicators:

✅ **Good Signs:**
- Cost loss steadily decreasing
- Agent reward trending upward (with variance)
- Best reward improving every 5-10 iterations
- Agent achieves >80% of expert reward

⚠️ **Warning Signs:**
- Cost loss oscillating wildly → reduce LR
- Agent reward collapsing → check cost function
- No improvement for >30 iterations → need different approach
- NaN losses → gradient explosion, add clipping

---

## Quick Tuning Guide

### If rewards are too low:
1. Increase `num_traj` (1000 → 2000)
2. Increase `horizon` (100 → 150)
3. Decrease `lambda_` (0.001 → 0.0001)

### If training is unstable:
1. Decrease learning rate (3e-4 → 1e-4)
2. Increase gradient clipping (1.0 → 0.5)
3. Reduce batch size (256 → 128)

### If converging too slowly:
1. Increase `hidden_dim` (128 → 256)
2. Increase `reward_fn_updates` (30 → 50)
3. Increase learning rate (3e-4 → 5e-4)

---

## Advanced: Curriculum Learning

Start easy, gradually increase difficulty:

```python
# Adjust these per iteration
if iteration < 50:
    horizon = 50
    num_traj = 500
elif iteration < 100:
    horizon = 75
    num_traj = 750
else:
    horizon = 100
    num_traj = 1000
```

---

## Comparison: GCL vs RGCL vs GAIL

| Method | Stability | Speed | Convergence | Best For |
|--------|-----------|-------|-------------|----------|
| **GCL** | ⚠️ Medium | ⚡ Fast | 🐢 Slow | Simple tasks |
| **RGCL** | ✅ High | ⚡ Medium | 🚀 Fast | Complex tasks, unstable GCL |
| **GAIL** | ✅ High | ⚡ Fast | 🏃 Medium | Large datasets |
| **AIRL** | ✅ Very High | 🐢 Slow | 🏃 Medium | Transfer learning |

## Quick Decision Tree

```
Is training unstable? (loss oscillating wildly)
├─ YES → Try RGCL first, then GAIL if still unstable
└─ NO → Is convergence too slow?
    ├─ YES → Try RGCL or increase MPPI samples
    └─ NO → Continue with current settings
```

## Final Notes

- **Simplified dynamics** are approximate - perfect match is unlikely
- **GCL is sensitive** to hyperparameters - expect to tune
- **RGCL adds stability** via second-order information
- **MPPI is expensive** - tradeoff speed vs accuracy
- **Consider alternatives**: GAIL, AIRL, BC (behavioral cloning)

**Recommended Starting Point for Walker2d:**
1. Try **RGCL** first (most stable)
2. If too slow, switch to **GCL** with careful tuning
3. If still unstable, try **GAIL**

Good luck! 🚀

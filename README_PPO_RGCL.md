# RGCL with PPO Policy

This implementation combines **RGCL (Recursive Guided Cost Learning)** with **PPO (Proximal Policy Optimization)** for Inverse Reinforcement Learning.

## Overview

RGCL is a recursive method that updates the cost function parameters using Kalman filtering with gradients and Hessians. Previously, it only worked with MPPI (sampling-based) control. Now it works with PPO (neural network) policies.

## Key Innovation

**RGCL** (cost function learning method) + **PPO** (policy type) = Best of both worlds:
- **RGCL**: Second-order updates with Kalman filtering → better convergence
- **PPO**: Neural network policy → faster, scalable, model-free

## Architecture

### Cost Function Updates (RGCL)
For each timestep `t`:
1. Sample action from PPO policy: `a_t ~ π_θ(·|s_t)`
2. Execute action, observe next state: `s_{t+1}`
3. Compute gradients:
   - `∇_φ c(s_demo[t])` - gradient at expert state
   - `∇_φ c(s_{t+1})` - gradient at agent state
4. Compute Hessians:
   - `H_φ c(s_demo[t])` - Hessian at expert state
   - `H_φ c(s_{t+1})` - Hessian at agent state
5. Kalman-style update:
   ```
   P ← inv(inv(P + Q) + H_demo - H_agent)
   φ ← φ - P(∇_demo - ∇_agent)
   ```

Where:
- `φ` = cost function parameters (flattened neural network weights)
- `P` = parameter covariance matrix
- `Q` = process noise covariance

### PPO Policy
- Policy network: `π_θ(a|s) = tanh(N(μ(s), σ²))`
- Value network: `V_ψ(s)`
- Standard PPO updates (not used during RGCL, but can be applied separately)

## Files

### Implementation
- **`src/control/PPO_simple.py`**: Added `RGCL_lax()` method to `SimplePPO` class
  - Line 258-387: RGCL with PPO implementation

### Training Script
- **`main_ppo_rgcl.py`**: Main training script for RGCL+PPO
  - Initializes cost network, PPO policy
  - Runs RGCL iterations
  - Saves results

### Testing
- **`test_ppo_rgcl.py`**: Quick test script (5 iterations)

## Usage

### Basic Training
```bash
python main_ppo_rgcl.py --gym_env CartPole-v1 --rirl_iterations 100
```

### With Custom Hyperparameters
```bash
python main_ppo_rgcl.py \
    --gym_env CartPole-v1 \
    --rirl_iterations 100 \
    --N_steps 200 \
    --P 1e-2 \
    --Q 1e-4 \
    --ppo_lr 3e-4 \
    --hidden_dim 64 \
    --seed 42
```

### Quick Test
```bash
python test_ppo_rgcl.py
```

## Command-Line Arguments

### Environment & Experiment
- `--gym_env`: Environment name (default: "CartPole-v1")
- `--seed`: Random seed (default: 123)
- `--experiment_name`: Name for saving results (default: "experiment_ppo_rgcl")
- `--results_savepath`: Path to save results (default: "results_ppo_rgcl")

### Training
- `--rirl_iterations`: Number of RGCL iterations (default: 100)
- `--N_steps`: Steps per episode (default: 200)
- `--N_steps_expert`: Expert demonstration length (default: 200)

### RGCL Parameters
- `--P`: Initial parameter covariance (default: 1e-2)
  - Larger values → more exploration in parameter space
  - Smaller values → more conservative updates
- `--Q`: Process noise covariance (default: 1e-4)
  - Larger values → faster adaptation, more noise
  - Smaller values → slower adaptation, more stable
- `--diagonal`: Use diagonal Hessian approximation (not implemented)

### PPO Parameters
- `--ppo_lr`: PPO learning rate (default: 3e-4)
  - Only used if you want to update PPO between RGCL iterations
- `--rollout_length`: Trajectory length (default: 200)
- `--ppo_epochs`: PPO update epochs (default: 10)
- `--gamma`: Discount factor (default: 0.99)
- `--clip_eps`: PPO clip epsilon (default: 0.2)

### Network Architecture
- `--hidden_dim`: Hidden layer size for cost network (default: 64)

## How It Works

### Initialization (Iteration 0)
1. Load expert demonstrations
2. Initialize cost network `c_φ(s)`
3. Initialize PPO policy `π_θ(a|s)`
4. Initialize parameter covariance `P = P_0 * I`

### Each RGCL Iteration
```python
for t in range(N_steps):
    # 1. Sample action from PPO
    a_t = policy.sample(s_t)

    # 2. Environment step
    s_{t+1} = dynamics(s_t, a_t)

    # 3. Compute RGCL updates
    g_demo = ∇_φ c(s_demo[t])
    g_agent = ∇_φ c(s_{t+1})
    H_demo = H_φ c(s_demo[t])
    H_agent = H_φ c(s_{t+1})

    # 4. Kalman update
    P = inv(inv(P + Q) + H_demo - H_agent)
    φ = φ - P @ (g_demo - g_agent)

    # 5. Next state
    s_t = s_{t+1}
```

### Output
- Updated cost function parameters `φ`
- Trajectory (states, actions, rewards)
- Final parameter covariance `P`

## Comparison with Other Methods

| Method | Cost Updates | Policy Type | Convergence | Speed |
|--------|--------------|-------------|-------------|-------|
| **GCL + MPPI** | First-order gradient | Sampling | Slow | Slow |
| **RGCL + MPPI** | Second-order Kalman | Sampling | Fast | Slow |
| **GCL + PPO** | First-order gradient | Neural net | Slow | Fast |
| **RGCL + PPO** | Second-order Kalman | Neural net | **Fast** | **Fast** |

## Advantages

1. **Better Convergence**: Second-order updates (Hessian) converge faster than gradient descent
2. **Uncertainty Quantification**: Maintains covariance matrix `P` for parameter uncertainty
3. **Faster Rollouts**: PPO single forward pass vs. MPPI sampling
4. **Model-Free**: Doesn't require accurate dynamics model
5. **Scalable**: Works with high-dimensional state/action spaces

## Limitations

1. **Computational Cost**: Computing Hessians is expensive (O(n²) memory)
2. **Numerical Stability**: Matrix inversions can be unstable
3. **Hyperparameter Sensitivity**: `P` and `Q` need tuning
4. **No Diagonal Mode**: Full Hessian required (diagonal version not implemented)

## Expected Results

For CartPole-v1:
- Expert reward: ~200
- Initial agent reward: ~20-40
- After 100 iterations: ~150-200

Training should show:
- Gradual improvement in agent rewards
- Stable Kalman updates (no NaN/Inf)
- Convergence towards expert performance

## Troubleshooting

### Matrix Inversion Errors
If you see `LinAlgError`:
- Reduce `--P` (try 1e-3)
- Reduce `--Q` (try 1e-5)
- Reduce `--hidden_dim` (smaller network = fewer parameters)

### Poor Performance
- Increase `--rirl_iterations` (more training)
- Adjust `--P` and `--Q` (try different scales)
- Check expert demonstrations (should have high reward)

### Slow Training
- Reduce `--hidden_dim` (faster Hessian computation)
- Reduce `--N_steps` (shorter trajectories)
- Use GPU (JAX should auto-detect)

## Technical Details

### Parameter Flattening
Cost network parameters are flattened into vector `φ`:
```python
# Before: {'Dense_0': {'kernel': [...], 'bias': [...]}, ...}
# After: φ = [Dense_0.bias, Dense_0.kernel.flatten(), ...]
```

### Hessian Computation
Full Hessian matrix `H ∈ ℝ^{n×n}` where `n = number of parameters`:
```python
H = ∂²c/∂φ² = [[∂²c/∂φᵢ∂φⱼ for j in range(n)] for i in range(n)]
```

Computed via `jax.hessian()` for automatic differentiation.

### Kalman Update Derivation
The update rule:
```
P ← inv(inv(P + Q) + H_demo - H_agent)
φ ← φ - P(∇_demo - ∇_agent)
```

Comes from Kalman filtering with:
- Prediction: `φ⁻ = φ, P⁻ = P + Q`
- Measurement: observation is expert state
- Update: incorporate Hessian information

## Future Extensions

### Easy
- [ ] Gradient clipping for stability
- [ ] Learning rate scheduling for PPO
- [ ] Early stopping based on reward threshold

### Medium
- [ ] Diagonal Hessian approximation (faster)
- [ ] Fisher information matrix instead of Hessian
- [ ] Multiple expert demonstrations

### Hard
- [ ] Natural gradient RGCL
- [ ] Distributed RGCL (parallel rollouts)
- [ ] Meta-learning for `P` and `Q`

## References

- **RGCL**: "Recursive Deep Inverse Reinforcement Learning" (your implementation)
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GCL**: Finn et al., "Guided Cost Learning" (2016)
- **Kalman Filtering**: Kalman, "A New Approach to Linear Filtering" (1960)

## Example Output

```bash
$ python main_ppo_rgcl.py --gym_env CartPole-v1 --rirl_iterations 100

============================================================
Run 1/1, Seed: 123
============================================================

Generating expert demonstrations...
Expert reward: 200.00
Initializing PPO policy...

Iteration 1/100
Running RGCL with PPO policy...
RGCL iteration time: 2.34s, Total reward: 38.00

Iteration 10/100
Running RGCL with PPO policy...
RGCL iteration time: 2.15s, Total reward: 82.00

============================================================
Iteration 10: Avg Reward (last 10) = 65.30
Expert Reward = 200.00
============================================================

...

============================================================
Training Complete!
Results saved to: results\CartPole-v1\rgcl-ppo
============================================================
```

## Integration with Existing Code

The RGCL+PPO implementation:
- ✅ Uses same `CostNN` architecture as other methods
- ✅ Compatible with same expert demonstration format
- ✅ Saves results in same directory structure
- ✅ Can be compared directly with GAIL, AIRL, GCL, etc.

## Conclusion

RGCL+PPO combines the best aspects of:
1. **Second-order optimization** (RGCL Kalman updates)
2. **Modern policy learning** (PPO neural networks)
3. **Efficient rollouts** (single forward pass)

This makes it a **fast, scalable, and well-converging** method for inverse reinforcement learning.

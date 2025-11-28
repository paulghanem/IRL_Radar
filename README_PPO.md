# IRL with PPO Policy - CartPole Implementation

This implementation replaces the MPPI (Model Predictive Path Integral) controller with PPO (Proximal Policy Optimization) for trajectory generation in Inverse Reinforcement Learning.

## Overview

Instead of using sampling-based MPPI control, this version uses a learned neural network policy (PPO) to generate trajectories. The cost function is still learned through IRL methods (GAIL, AIRL, GCL, etc.), but the policy is a trainable neural network.

## Key Differences from MPPI Version

| Aspect | MPPI Version | PPO Version |
|--------|--------------|-------------|
| **Policy** | Sampling-based trajectory optimization | Learned neural network policy |
| **Control** | Requires dynamics model | Model-free (learns from experience) |
| **Computation** | Samples many trajectories per step | Single trajectory rollout |
| **Learning** | Only learns cost function | Learns both policy and cost function |
| **Speed** | Slower (many samples) | Faster (single policy forward pass) |

## Files Created

### Core Implementation
- **`src/control/PPO_simple.py`**: Simplified PPO implementation for CartPole and simple environments
  - `SimplePPO`: Main PPO class with policy rollout and updates
  - `PolicyModel`: Gaussian policy network (continuous actions)
  - `CriticModel`: Value function network

### Training Scripts
- **`main_ppo.py`**: Main training script for IRL with PPO
  - Supports all IRL methods: GCL, GAIL, AIRL, SQIL, Upper Bound
  - Uses PPO for trajectory generation instead of MPPI

### Testing Scripts
- **`test_ppo_cartpole.py`**: Unit test for PPO implementation
  - Tests policy rollout
  - Tests PPO updates
  - Validates shapes and functionality

- **`run_ppo_test.py`**: Quick integration test
  - Runs full IRL training with PPO
  - Uses reduced iterations for faster testing

## How to Run

### 1. Test PPO Implementation

```bash
python test_ppo_cartpole.py
```

Expected output:
```
[OK] Rollout successful!
  States shape: (200, 4)
  Actions shape: (200, 1)
  Total reward: 40.00
  Average reward: 0.2000
[OK] PPO update successful!
All tests passed! [OK]
```

### 2. Run Quick Test (20 iterations)

```bash
python run_ppo_test.py
```

This runs a short training session to verify everything works.

### 3. Run Full Training

```bash
# Basic GCL with PPO
python main_ppo.py --gym_env CartPole-v1 --rirl_iterations 100

# GAIL with PPO
python main_ppo.py --gym_env CartPole-v1 --rirl_iterations 100 --gail

# AIRL with PPO
python main_ppo.py --gym_env CartPole-v1 --rirl_iterations 100 --airl

# With custom hyperparameters
python main_ppo.py \
    --gym_env CartPole-v1 \
    --rirl_iterations 100 \
    --N_steps 200 \
    --reward_fn_updates 10 \
    --lr 1e-3 \
    --ppo_lr 3e-4 \
    --rollout_length 200 \
    --ppo_epochs 10 \
    --hidden_dim 64 \
    --seed 42
```

## Command-Line Arguments

### Environment & Experiment
- `--gym_env`: Environment name (default: "CartPole-v1")
- `--seed`: Random seed (default: 123)
- `--experiment_name`: Name for saving results (default: "experiment_ppo")
- `--results_savepath`: Path to save results (default: "results_ppo")

### Training
- `--rirl_iterations`: Number of training iterations (default: 100)
- `--N_steps`: Steps per episode for agent (default: 200)
- `--N_steps_expert`: Steps per episode for expert (default: 200)
- `--reward_fn_updates`: Cost function updates per iteration (default: 10)

### Learning Rates
- `--lr`: Learning rate for cost function (default: 1e-3)
- `--ppo_lr`: Learning rate for PPO policy (default: 3e-4)

### PPO Hyperparameters
- `--rollout_length`: PPO trajectory length (default: 200)
- `--ppo_epochs`: PPO update epochs (default: 10)
- `--ppo_batch_size`: PPO minibatch size (default: 64)
- `--gamma`: Discount factor (default: 0.99)
- `--clip_eps`: PPO clip epsilon (default: 0.2)

### Network Architecture
- `--hidden_dim`: Hidden layer size for cost network (default: 64)

### IRL Methods
- `--gail`: Use GAIL method
- `--airl`: Use AIRL method
- `--sqil`: Use SQIL method
- `--UB`: Use Upper Bound loss

## Architecture

### PPO Policy Network

```python
PolicyModel (64 → tanh → 64 → tanh → action_dim)
├── Mean (μ): Linear output
└── Log-Std (log σ): Learnable parameter

Action ~ tanh(N(μ, exp(log σ)))
```

### Value Network

```python
CriticModel (64 → tanh → 64 → tanh → 1)
└── Value estimate V(s)
```

### Cost Network (learned via IRL)

```python
CostNN (hidden_dim → ReLU → hidden_dim → ReLU → 1)
└── Cost c(s) clipped to [0, 5]
```

## Training Loop

Each iteration:

1. **Generate Trajectory**: Run PPO policy for `rollout_length` steps
2. **Update Cost Function**: Train cost network to distinguish expert from agent
   - Uses IRL loss (GCL, GAIL, AIRL, etc.)
   - Updates for `reward_fn_updates` steps
3. **Update PPO Policy**: Improve policy using learned cost as negative reward
   - Compute advantages with GAE
   - Update policy with PPO clipping
   - Update value function
4. **Track Progress**: Log rewards and losses

## Results

Training saves:
- `results/<env>/<method>/cost_seed=<seed>.npy`: Agent rewards over time
- `results/<env>/<method>/expert_cost_seed=<seed>.npy`: Expert rewards
- `results/<env>/<method>/loss_seed=<seed>.npy`: Cost function losses

## Example Output

```
Iteration 1/100
Generating trajectories with PPO...
Rollout time: 0.4s, Total reward: 42.00
Updating cost function...
Cost function loss: -1.234
Updating PPO policy...

Iteration 10/100
...
============================================================
Iteration 10: Avg Reward (last 10) = 45.30
Expert Reward = 200.00
============================================================
```

## Advantages of PPO over MPPI

1. **Faster**: Single forward pass vs. many trajectory samples
2. **Model-free**: Doesn't require accurate dynamics model
3. **Scalable**: Works with high-dimensional action spaces
4. **Sample efficient**: Learns from experience, improves over time
5. **Smooth policies**: Neural network provides smooth action mapping

## Limitations

1. **Requires more training**: PPO needs to learn policy from scratch
2. **Hyperparameter sensitive**: Learning rates, clip epsilon need tuning
3. **Can be unstable**: May need careful initialization
4. **Local optima**: Gradient-based, can get stuck

## Troubleshooting

### NaN Losses
If you see `Cost function loss: nan`:
- Reduce learning rate `--lr`
- Reduce `--reward_fn_updates`
- Add gradient clipping (modify code)

### Poor Performance
- Increase `--ppo_epochs` (more policy updates)
- Increase `--rollout_length` (more experience per iteration)
- Adjust `--ppo_lr` (policy learning rate)
- Try different `--seed` values

### Memory Issues
- Reduce `--rollout_length`
- Reduce `--ppo_batch_size`
- Reduce `--hidden_dim`

## Next Steps

To extend this implementation:

1. **Add more environments**: Extend `PPO_simple.py` reward_fn for Pendulum, MountainCar
2. **Add recurrent policies**: Use LSTM in PolicyModel
3. **Add curiosity**: Intrinsic motivation for exploration
4. **Add multi-task**: Train on multiple environments simultaneously
5. **Add TRPO**: Replace PPO with Trust Region Policy Optimization

## References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAIL**: Ho and Ermon, "Generative Adversarial Imitation Learning" (2016)
- **AIRL**: Fu et al., "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" (2017)
- **GCL**: Finn et al., "Guided Cost Learning" (2016)

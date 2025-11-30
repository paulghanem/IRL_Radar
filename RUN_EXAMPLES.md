# How to Run IRL with Integrated MuJoCo Experts

All MuJoCo experts (Walker2d, Hopper, HalfCheetah, Swimmer) are now properly integrated!

## Verified Environments

All environments have been tested and verified:
- **Walker2d**: State dim 17, Action dim 6 ✓
- **Hopper**: State dim 11, Action dim 3 ✓
- **HalfCheetah-v4**: State dim 17, Action dim 6 ✓
- **Swimmer**: State dim 8, Action dim 2 ✓

## Usage Examples

### 1. Test Experts (Quick Verification)

```bash
python verify_experts.py
```

### 2. Run IRL with RGCL (Recommended)

#### Walker2d
```bash
python main.py --gym_env Walker2d --rgcl --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name walker2d_rgcl --seed 123
```

#### Hopper
```bash
python main.py --gym_env Hopper --rgcl --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name hopper_rgcl --seed 123
```

#### HalfCheetah
```bash
python main.py --gym_env HalfCheetah-v4 --rgcl --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name halfcheetah_rgcl --seed 123
```

#### Swimmer
```bash
python main.py --gym_env Swimmer --rgcl --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name swimmer_rgcl --seed 123
```

### 3. Run IRL with GAIL

```bash
python main.py --gym_env Walker2d --gail --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name walker2d_gail --seed 123
```

### 4. Run IRL with AIRL

```bash
python main.py --gym_env Walker2d --airl --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name walker2d_airl --seed 123
```

### 5. Run with PPO Policy (instead of MPPI)

```bash
python main.py --gym_env Walker2d --rgcl --PPO --N_steps 1000 --N_steps_expert 1000 --rirl_iterations 100 --experiment_name walker2d_rgcl_ppo --seed 123
```

## Key Parameters

- `--gym_env`: Environment name (Walker2d, Hopper, HalfCheetah-v4, Swimmer)
- `--rgcl`: Use RGCL method (default, recommended)
- `--gail`: Use GAIL method
- `--airl`: Use AIRL method
- `--PPO`: Use PPO policy instead of MPPI
- `--N_steps`: Number of steps per trajectory
- `--N_steps_expert`: Number of expert demonstration steps
- `--rirl_iterations`: Number of IRL iterations
- `--horizon`: MPPI horizon (default: 50)
- `--num_traj`: Number of MPPI trajectories (default: 2000)
- `--lr`: Learning rate (default: 1e-4)
- `--lambda_`: MPPI temperature (default: 0.01)
- `--seed`: Random seed

## Output

Results are saved to:
- `results/<experiment_name>_<seed>/`
- Cost plots: `results/<env_name>/<method>/cost_*.npy`
- Expert cost: `results/<env_name>/<method>/expert_cost_*.npy`

## Integration Details

The integration works as follows:

1. **Expert Loading** (utils/helpers.py):
   - GenerateDemo class automatically detects environment type
   - Loads TD3 experts from `experts/` folder
   - Handles observation space matching (v3 experts with v4 environments)

2. **Environment Configuration** (main.py lines 213-243):
   - Each MuJoCo environment has proper XML model, frame_skip, and dt settings
   - MJX (MuJoCo XLA) model is created for fast simulation

3. **IRL Training** (main.py lines 298-575):
   - Expert demonstrations are generated once at start
   - Cost function is learned via RGCL/GAIL/AIRL
   - Policy (MPPI or PPO) uses learned cost to generate trajectories
   - Results are saved every 10 iterations

## Troubleshooting

If you get observation space errors:
- Ensure you're using the environment names exactly as shown above
- The code automatically handles v3/v4 compatibility

If experts fail to load:
- Check that files exist in `experts/` folder:
  - td3-Walker2d-v3.zip
  - td3-Hopper-v3.zip
  - td3-HalfCheetah-v3.zip
  - td3-Swimmer-v4.zip

# State vs Observation Mismatch - Analysis and Fix

## Problem Identified

### The Mismatch

**Expert Demonstrations (helpers.py line 269)**:
- Saves `obs.ravel()` - the **observation** from the environment
- When `exclude_current_positions_from_observation=True`:
  - Walker2d: 17 dims (no x-position)
  - Hopper: 11 dims (no x-position)
  - HalfCheetah: 17 dims (no x-position)

**MPPI Dynamics (dynamics.py line 342)**:
- Returns `jnp.concatenate([carry_data.qpos, carry_data.qvel])` - the **full state**
- Includes ALL qpos elements (including x-position):
  - Walker2d: 18 dims (qpos[9] + qvel[9], x-position at index 0)
  - Hopper: 12 dims (qpos[6] + qvel[6], x-position at index 0)
  - HalfCheetah: 18 dims (qpos[9] + qvel[9], x-position at index 0)

**MPPI Reward Calculation (mppi_class.py line 550)**:
```python
forward_reward=(next_state[0]-state[0])/(dt*frame_skip)
```
- Expects `state[0]` to be x-position
- But expert demos have `state[0]` as first observation element (NOT x-position)

### Impact

1. **Dimension mismatch**: Expert demos are 17-dim, MPPI generates 18-dim states
2. **Incorrect reward calculation**: Expert demos don't have x-position at state[0]
3. **IRL training failure**: Cost function learns on mismatched state spaces

## MuJoCo Environment State Structure

### Walker2d-v4
- **qpos** (9): [x_pos, z_pos, orientation, thigh_joint, leg_joint, foot_joint, ...]
- **qvel** (9): [x_vel, z_vel, ang_vel, ...]
- **Observation (exclude_x=True)** (17): [z_pos, orientation, ..., x_vel, z_vel, ...]
- **Full State** (18): [qpos (9) + qvel (9)]

### Hopper-v4
- **qpos** (6): [x_pos, z_pos, orientation, thigh_joint, leg_joint, foot_joint]
- **qvel** (6): [x_vel, z_vel, ang_vel, ...]
- **Observation (exclude_x=True)** (11): [z_pos, orientation, ..., x_vel, z_vel, ...]
- **Full State** (12): [qpos (6) + qvel (6)]

### HalfCheetah-v4
- **qpos** (9): [x_pos, angle, ...]
- **qvel** (9): [x_vel, ang_vel, ...]
- **Observation (exclude_x=True)** (17): [angle, ..., x_vel, ang_vel, ...]
- **Full State** (18): [qpos (9) + qvel (9)]

### Swimmer-v4
- **qpos** (5): [x_pos, y_pos, orientation, ...]
- **qvel** (5): [x_vel, y_vel, ang_vel, ...]
- **Observation** (8): Varies by environment
- **Full State** (10): [qpos (5) + qvel (5)]

## Solution

We need to extract and save the **full state** (qpos + qvel) from the MuJoCo environment during expert demonstration collection, not just the observation.

### Option 1: Access underlying MuJoCo data
```python
# In helpers.py, after vec_env.step(action)
# Access the underlying gym environment
gym_env = vec_env.envs[0].unwrapped
# Get full state
qpos = gym_env.data.qpos.copy()
qvel = gym_env.data.qvel.copy()
full_state = np.concatenate([qpos, qvel])
state_seq.append(full_state)
```

### Option 2: Use info dict
Some Gymnasium environments provide full state in the info dict:
```python
obs, reward, done, info = vec_env.step(action)
if 'x_position' in info:
    x_pos = info['x_position']
    full_state = np.concatenate([[x_pos], obs])
```

### Option 3: Modify environment wrapper
Create a wrapper that always returns full state alongside observation.

## Recommended Fix

Modify `utils/helpers.py` to extract full MuJoCo state:

```python
def generate_gymnasium_demo(self, env_name, max_frames=200, seed=123):
    # ... existing code ...

    # Access underlying environment
    base_env = vec_env.envs[0].unwrapped

    while True:
        # Get observation for policy
        state_seq.append(obs.ravel())

        # Get FULL state for dynamics/rewards
        if hasattr(base_env, 'data'):  # MuJoCo environment
            qpos = base_env.data.qpos.copy()
            qvel = base_env.data.qvel.copy()
            full_state = np.concatenate([qpos, qvel])
            full_state_seq.append(full_state)

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # ... rest of code ...

    return np.stack(state_seq), np.stack(action_seq), rewards, vec_env, np.stack(full_state_seq)
```

Then in main.py, use `full_state_seq` for MPPI dynamics and reward calculation.

# Simplified Walker2d Test - Pure JAX vs MJX Comparison

## Summary

Testing whether **pure JAX analytical dynamics** are faster than **MJX physics simulation** for IRL experiments.

Created simplified Walker2d environment with same dimensions as real Walker2d (17 state, 6 action) but using analytical dynamics instead of MJX.

---

## Test Configuration

### Environment: Simplified Walker2d (Pure JAX)

**State (17-dim):**
```
[0]: x position (rootx)
[1]: z height
[2]: body angle
[3-8]: joint angles (6)
[9]: x velocity
[10]: z velocity
[11]: angular velocity
[12-17]: joint velocities (6)
```

**Action (6-dim):** Joint torques

**Dynamics:** Analytical (no MJX calls)
- Joint accelerations from actions
- Joint motion affects body motion
- Gravity and stability constraints
- All operations in pure JAX (JIT-compiled)

**Reward Function:**
- Forward velocity reward
- Alive bonus (stay upright)
- Control cost penalty

### Experiment Parameters

Both GCL and RGCL LAX tested with:
- **Horizon:** 50
- **Num trajectories:** 500
- **N_steps:** 1000
- **Iterations:** 1000
- **RGCL Q:** 1e-5
- **RGCL P:** 1e-2

---

## Jobs Submitted

### Job 36240199: GCL with Simplified Walker2d
- **Status:** Pending (Priority)
- **Time limit:** 48 hours
- **Output:** `simple_walker2d_gcl_36240199.out`
- **Results:** `simple_walker2d_gcl_results.txt`

**Tests:**
- 1000 iterations of GCL
- Uses `generate_session_lax()` (original non-JIT function)
- Reports execution time and rewards

### Job 36240201: RGCL LAX with Simplified Walker2d
- **Status:** Pending
- **Time limit:** 48 hours
- **Output:** `simple_walker2d_rgcl_36240201.out`
- **Results:** `simple_walker2d_rgcl_results.txt`

**Tests:**
- 1000 iterations of RGCL LAX
- Uses `RGCL_lax()` (original non-JIT function)
- Reports execution time and rewards

---

## Expected Results

### Hypothesis
Pure JAX analytical dynamics may be faster because:
- No MJX physics engine overhead
- Simpler dynamics (fewer computations)
- Direct JAX operations (already JIT-compiled)
- No need for complex contact/constraint solving

### Performance Comparison

**Baseline (MJX Walker2d, horizon=50, num_traj=500):**
- Time per step: ~5 seconds (without JIT)
- Time per step: ~0.4-0.6 seconds (with JIT)
- 1000 iterations: ~11-16 hours (with JIT)

**Expected (Simple Walker2d):**
- If faster: < 0.4 seconds/step
- If similar: ~0.4-0.6 seconds/step
- If slower: > 0.6 seconds/step

### Results Will Show

1. **Execution time per iteration**
   - How fast is pure JAX vs MJX?
   - Is analytical dynamics worth the simplification?

2. **Total time for 1000 iterations**
   - Can we fit in 48-hour GPU window?
   - Comparison to MJX-based experiments

3. **Reward quality**
   - Do simplified dynamics produce meaningful rewards?
   - Can IRL methods learn from simplified environment?

---

## Implementation Files

### `src/control/simple_walker2d.py`
Contains pure JAX implementation:
```python
@jit
def simple_walker2d_step(state, action):
    """17-dim state -> 17-dim next state"""
    # Simplified dynamics (no MJX)

@jit
def simple_walker2d_reward(state, action, next_state):
    """Walker2d-style reward function"""
    # Forward velocity + alive bonus - control cost

@jit
def simple_walker2d_reset():
    """Reset to initial state"""
    # Standing position
```

### `test_simple_walker2d.py`
Comprehensive test script:
- Integrates simple Walker2d with MPPI
- Tests both GCL and RGCL LAX methods
- Reports timing and rewards
- Saves iteration-by-iteration data

### Job Scripts
- `run_simple_walker2d_gcl.sh` - GCL test
- `run_simple_walker2d_rgcl.sh` - RGCL LAX test

---

## How to Check Results

### Monitor Job Status
```bash
squeue -j 36240199,36240201
```

### View Real-Time Output
```bash
# GCL
tail -f simple_walker2d_gcl_36240199.out

# RGCL
tail -f simple_walker2d_rgcl_36240201.out
```

### Check Final Results
```bash
# GCL results
cat simple_walker2d_gcl_results.txt

# RGCL results
cat simple_walker2d_rgcl_results.txt
```

### Expected Output Format
```
Method: GCL
Parameters: horizon=50, num_traj=500
N_steps=1000, iterations=1000

Total time: X.XX seconds (Y.YY hours)
Average time per iteration: X.XXXX seconds
Average reward (all iterations): X.XX
Final reward (last 100 iterations): X.XX
Time per step (avg): X.XXXX seconds

Iteration-by-iteration data:
1,0.5234,12.45
2,0.5123,13.67
...
```

---

## Key Questions to Answer

1. **Speed Comparison:**
   - Is pure JAX faster than MJX?
   - By how much?
   - Worth the simplification?

2. **Learning Quality:**
   - Do IRL methods work with simplified dynamics?
   - Are rewards meaningful?
   - Can we trust the learned policies?

3. **Scalability:**
   - Can we run 1000 iterations in 48 hours?
   - Better than MJX for large-scale experiments?

4. **Trade-offs:**
   - Speed gain vs realism loss
   - When to use simplified vs full physics?

---

## Next Steps

1. ⏳ Wait for GPU allocation (jobs pending)
2. 📊 Monitor execution progress
3. 📈 Analyze results when complete
4. 🔍 Compare to MJX-based experiments
5. ✅ Decide which approach to use for future experiments

---

## Comparison Table (To Be Filled)

| Metric | MJX Walker2d (JIT) | Simple Walker2d | Winner |
|--------|-------------------|-----------------|--------|
| Time/step | ~0.4-0.6s | TBD | TBD |
| Time/1000 iter | ~11-16 hrs | TBD | TBD |
| Avg reward | TBD | TBD | TBD |
| Complexity | High (full physics) | Low (analytical) | Simple |
| Realism | High | Medium | MJX |
| Speedup | 8-15x (with JIT) | TBD | TBD |

---

**Status:** Jobs submitted and pending GPU allocation
**Created:** 2025-11-27
**Jobs:** 36240199 (GCL), 36240201 (RGCL)
**Files:**
- `src/control/simple_walker2d.py`
- `test_simple_walker2d.py`
- `run_simple_walker2d_gcl.sh`
- `run_simple_walker2d_rgcl.sh`

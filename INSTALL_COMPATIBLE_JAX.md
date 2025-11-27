# Fixing JAX cuDNN Compatibility Issue

## Problem
Your current JAX installation (0.6.2) requires cuDNN 9.8.0, but your system has cuDNN 9.1.0, causing this error:
```
FAILED_PRECONDITION: DNN library initialization failed
```

## Solution: Install Compatible JAX Version

Run these commands to install a JAX version compatible with your system's cuDNN:

```bash
# Load your environment
module load anaconda3/2024.10-1
conda activate rirl

# Uninstall current JAX
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt

# Install JAX 0.4.20 which is compatible with cuDNN 9.1.0
pip install --upgrade "jax[cuda12_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test the installation
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
```

## Alternative: Use the automated script

```bash
chmod +x fix_jax_install.sh
./fix_jax_install.sh
```

## After Installation

Once JAX is properly installed, you can run the benchmarks:

```bash
# Set required environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_DISABLE_X64=1

# Run the Brax vs MJX benchmark
python benchmark_brax_vs_mjx.py
```

## Note on Existing Changes

I've already made these code improvements for GPU compatibility:
1. Fixed `.clone()` calls in `src/control/mppi_class.py` (changed from PyTorch syntax to JAX)
2. Made x64 mode optional in `src/control/dynamics.py` (can be disabled with `JAX_DISABLE_X64=1`)

These changes ensure the code is GPU-compatible once JAX is properly installed.

#!/bin/bash
# Automated script to install JAX compatible with cuDNN 9.1.0

echo "========================================================================"
echo "Installing JAX compatible with cuDNN 9.1.0"
echo "========================================================================"
echo ""

# Load environment
module load anaconda3/2024.10-1
source activate rirl

echo "Current JAX installation:"
pip list | grep jax
echo ""

echo "Uninstalling current JAX packages..."
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt

echo ""
echo "Installing JAX 0.4.20 (compatible with cuDNN 9.1.0)..."
pip install --upgrade "jax[cuda12_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""

echo "Testing JAX installation..."
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "
import jax
import jax.numpy as jnp
print('✓ JAX version:', jax.__version__)
print('✓ Devices:', jax.devices())
print('✓ Backend:', jax.default_backend())

# Test basic GPU operation
x = jnp.ones(10)
y = x + 1
y.block_until_ready()
print('✓ Basic GPU operations working')
"

echo ""
echo "========================================================================"
echo "Setup complete! You can now run benchmarks with:"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  export JAX_DISABLE_X64=1"
echo "  python benchmark_brax_vs_mjx.py"
echo "========================================================================"

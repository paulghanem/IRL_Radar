#!/bin/bash
# Script to reinstall JAX with compatible cuDNN version

echo "=========================================="
echo "Fixing JAX cuDNN compatibility"
echo "=========================================="

# Load environment
module load anaconda3/2024.10-1
source activate rirl

echo "Current JAX version:"
python -c "import jax; print(f'JAX: {jax.__version__}')"

echo ""
echo "Reinstalling JAX with CUDA 12.1 support (compatible with cuDNN 9.1)..."
echo ""

# Uninstall current JAX
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt

# Install JAX with CUDA 12.1 support (this should use cuDNN 9.1)
# Using an older JAX version that's known to work with cuDNN 9.1
pip install --upgrade "jax[cuda12]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="

echo ""
echo "Testing JAX..."
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}'); print(f'Backend: {jax.default_backend()}')"

echo ""
echo "If you see GPU device above, the fix was successful!"
echo "=========================================="

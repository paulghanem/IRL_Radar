#!/bin/bash
# Wrapper script to run GPU test with proper CUDA configuration

set -e

echo "=== GPU Test Wrapper ===" 
echo "Node: $(hostname)"
echo ""

# Load required modules
echo "Loading modules..."
module load anaconda3/2024.10-1
module load cuda/12.4.0
echo "CUDA module loaded"

# Activate conda environment
echo "Activating conda environment..."
conda activate rirl

# Build LD_LIBRARY_PATH with all nvidia library directories from site-packages
echo "Configuring LD_LIBRARY_PATH..."
NVIDIA_SITE_PACKAGES=$(python -c "import sys, os; paths = [p for p in sys.path if 'site-packages' in p and os.path.exists(os.path.join(p, 'nvidia'))]; print(paths[0] if paths else '')")

if [ -n "$NVIDIA_SITE_PACKAGES" ]; then
    echo "Found nvidia in site-packages: $NVIDIA_SITE_PACKAGES/nvidia"
    NVIDIA_LIB_PATHS=$(find "$NVIDIA_SITE_PACKAGES/nvidia" -maxdepth 2 -name 'lib' -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"
    echo "Added nvidia library paths to LD_LIBRARY_PATH"
else
    echo "WARNING: Could not find nvidia in site-packages"
fi

# Configure JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo ""
echo "Environment configured. Running test..."
echo "=========================================="
echo ""

# Run the test
cd /ocean/projects/cis250114p/pghanem/IRL_Radar_big
python test_gpu_quick.py

echo ""
echo "=========================================="
echo "Test complete"

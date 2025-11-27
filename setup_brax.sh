#!/bin/bash
# Setup script for Brax with compatible dependencies

echo "========================================="
echo "Setting up Brax for IRL_Radar"
echo "========================================="

# Load anaconda module
echo "Loading anaconda module..."
module load anaconda3/2024.10-1

# Activate conda environment
echo "Activating rirl environment..."
conda activate rirl

# Check current JAX version
echo ""
echo "Current JAX version:"
python -c "import jax; print(f'  JAX: {jax.__version__}')" 2>/dev/null || echo "  JAX not found"

# Install Brax with compatible dependencies
echo ""
echo "Installing Brax (this may take a few minutes)..."
echo "Note: This will upgrade JAX to 0.6.x which Brax requires"

pip install brax==0.13.0 --no-deps
pip install dm-env pytinyrenderer

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import jax
from brax import envs
print(f'✓ JAX version: {jax.__version__}')
print(f'✓ JAX devices: {jax.devices()}')
print('✓ Brax imported successfully')
try:
    env = envs.get_environment('halfcheetah')
    print('✓ Brax HalfCheetah environment created')
except Exception as e:
    print(f'✗ Warning: Could not create environment: {e}')
    print('  This may be due to JAX version compatibility')
    print('  Try: pip install --force-reinstall jax==0.4.38 jaxlib==0.4.38')
" 2>&1

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "To use Brax in your experiments:"
echo "  1. Run: source setup_brax.sh"
echo "  2. Run your experiments normally"
echo ""
echo "To benchmark Brax vs MJX:"
echo "  python benchmark_brax_vs_mjx.py"
echo "========================================="

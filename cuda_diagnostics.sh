#!/bin/bash
module load anaconda3/2024.10-1
conda activate rirl

python << 'PYEOF'
import os
import sys
import glob

print("=" * 60)
print("CUDA DIAGNOSTICS")
print("=" * 60)

# Check conda environment
conda_prefix = os.environ.get('CONDA_PREFIX', 'NOT SET')
print(f"\n1. Conda Environment: {conda_prefix}")

# Check CUDA-related environment variables
print("\n2. Environment Variables:")
for var in ['XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'LD_LIBRARY_PATH', 
            'CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']:
    value = os.environ.get(var, 'NOT SET')
    if var == 'LD_LIBRARY_PATH' and value != 'NOT SET' and len(value) > 100:
        print(f"   {var}: {value[:100]}...")
    else:
        print(f"   {var}: {value}")

# Check for CUDA libraries in conda environment
print("\n3. CUDA Libraries in Conda:")
if conda_prefix != 'NOT SET':
    cuda_libs = glob.glob(f"{conda_prefix}/lib/libcuda*")
    cudnn_libs = glob.glob(f"{conda_prefix}/lib/libcudnn*")
    cublas_libs = glob.glob(f"{conda_prefix}/lib/libcublas*")
    
    print(f"   libcuda*: {len(cuda_libs)} files")
    for lib in cuda_libs[:3]:
        print(f"     - {os.path.basename(lib)}")
    
    print(f"   libcudnn*: {len(cudnn_libs)} files")
    for lib in cudnn_libs[:3]:
        print(f"     - {os.path.basename(lib)}")
    
    print(f"   libcublas*: {len(cublas_libs)} files")
    for lib in cublas_libs[:3]:
        print(f"     - {os.path.basename(lib)}")

# Check JAX installation
print("\n4. JAX Installation:")
try:
    import jax
    print(f"   JAX version: {jax.__version__}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Try to initialize JAX with minimal configuration
print("\n5. JAX Device Detection (minimal config):")
try:
    devices = jax.devices()
    print(f"   JAX devices: {devices}")
    for i, dev in enumerate(devices):
        print(f"     Device {i}: {dev.platform} - {dev.device_kind}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 60)
PYEOF

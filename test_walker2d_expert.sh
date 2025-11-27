#!/bin/bash

echo "Testing Walker2d Expert (6.4M steps checkpoint)"
echo "Running 1000 rollout steps..."
echo ""

module load anaconda3/2024.10-1
source activate rirl

python3 test_walker2d_expert.py

#!/bin/bash

for seed in {123..126}
do
  sbatch --export=SEED=$seed gpu_sbatch
done


#!/bin/bash

for seed in {130..134}
do
  sbatch --export=SEED=$seed gpu_sbatch
done


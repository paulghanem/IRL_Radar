#!/bin/bash

for seed in {129..134}
do
  sbatch --export=SEED=$seed gpu_sbatch
done


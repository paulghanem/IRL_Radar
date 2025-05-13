#!/bin/bash

for seed in {123..134}
do
  sbatch --export=SEED=$seed gpu_sbatch
done


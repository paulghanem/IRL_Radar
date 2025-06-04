# Recursive Deep Inverse Reinforcement Learning (RDIRL)
This repo is a working implementation of Recursive Deep Inverse Reinforcement Learning, Currently, it supports the `CartPole-v1` and `MountainCarContinuous-v0` environments.

## Create virtual environment with python 3.10.16
```bash
conda create --name rirl python==3.10.16
```
## How to install required packages

```bash
pip install requirements.txt
```

## Test if required packages are installed

```bash
python test.py
```

## How to run:

```bash
$ python main.py --horizon=85 --N_steps=150 --gym_env=MountainCarContinuous-v0 --lr=1e-4 --num_traj=3500 --reward_fn_updates=15 --lambda_=0.001 --rirl_iterations=10 --rgcl                    
$ python main.py --horizon=50 --N_steps=150 --gym_env=CartPole-v1 --lr=1e-4 --num_traj=2000 --reward_fn_updates=15 --lambda_=0.01 --rirl_iterations=10 --rgcl         
$ python main.py --horizon=50 --N_steps=200 --gym_env=Walker2d --lr=1e-4 --num_traj=500 --reward_fn_updates=15 --lambda_=0.01 --rirl_iterations=100 --rgcl                          
$ python main.py --horizon=50 --N_steps=200 --gym_env=HalfCheetah-v4 --lr=1e-4 --num_traj=500 --reward_fn_updates=15 --lambda_=0.01 --rirl_iterations=100 --rgcl                    
```
- use flags:  <br />
 --rgcl for RDIRL <br />
 --airl for AIRL <br />
 --gail for GAIL <br />
 --sqil for SQIL <br />
 --ub for Upper bound <br />
   no flag for GCL
 --online for online adpatation of all methods <br />
 
## Description of files:
- [generating_expert.py](generating_expert.py): Generates an expert on CartPole, by training vanilla policy gradient, and finally stores trained trajecteries as expert samples at [expert_agents](expert_agents).
- [experts/PG.py](experts/PG.py): Implementation of vanilla policy gradient. This is reused at several places.
- [generate_PPO.py](generate_PPO.py): Generates an expert on Mujoco Environments and Mountain car, by training vanilla policy gradient, and finally stores trained trajecteries as expert samples at [expert_agents](expert_agents).
- [main.py](main.py): Implementation of RDIRL and other benchmarking methods, main file to be ran.


## Results:
-[results/](results/): resulting reward functions <br />

# Recursive Deep Inverse Reinforcement Learning (RDIRL)
This repo is a working implementation of Recursive Deep Inverse Reinforcement Learning, Currently, it supports the `CartPole-v1` and `MountainCarContinuous-v0` environments.

## How to run:

```bash
$ python main.py --horizon=85 --N_steps=150 --gym_env=MountainCarContinuous-v0 --lr=1e-4 --num_traj=3500 --reward_fn_updates=15 --lambda_=0.001 --rirl_iterations=10 --rgcl                    
$ python main.py --horizon=50 --N_steps=150 --gym_env=CartPole-v1 --lr=1e-4 --num_traj=2000 --reward_fn_updates=15 --lambda_=0.01 --rirl_iterations=10 --rgcl                          
```
- use flags:  <br />
 --rgcl for RDIRL <br />
 --airl for GAN-GCL <br />
 --gail for GAIL <br />
   no flag for GCL
## Description of files:
- [generating_expert.py](generating_expert.py): Generates an expert on CartPole, by training vanilla policy gradient, and finally stores trained trajecteries as expert samples at [expert_samples](expert_samples).
- [experts/PG.py](experts/PG.py): Implementation of vanilla policy gradient. This is reused at several places.
- [main.py](main.py): Implementation of RDIRL and other benchmarking methods, main file to be ran.


## Results:
-[results/plotting/](results/plotting/): resulting reward functions <br />
-[per_episode_reward_IRL_gym.pdf](per_episode_reward_IRL_gym.pdf): resulting plots


## How to install required packages

```bash
pip install requirements.txt
```

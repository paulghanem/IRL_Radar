from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .rgcl import RGCL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'rgcl': RGCL
}

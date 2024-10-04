from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from envs.lbfenv import LBFEnvWrapper
import sys
import os

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}
REGISTRY["lbf"] = partial(env_fn, env=LBFEnvWrapper)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

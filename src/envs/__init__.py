from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from envs.lbfenv import LBFEnvWrapper
from envs.foragingenv import ForagingEnv
import sys
import os

def env_fn(env, **kwargs):
    print(f' Kwargs: {kwargs}' )
    if isinstance(env, str):
        # Check for the specific environment type and return the correct callable
        if env == "lbf":
            return LBFEnvWrapper(**kwargs)  # Create the environment with any additional parameters
        else:
            raise ValueError(f"Unknown environment: {env}")
    return env(**kwargs)

REGISTRY = {}
REGISTRY["lbf"] = partial(env_fn, env=LBFEnvWrapper)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

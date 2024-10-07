import lbforaging
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class LBFEnvWrapper(object):
    def __init__(self, env_name="Foraging-8x8-2p-1f-v3", seed=None):
        self.env = gym.make(env_name)
        self.n_agents = self.env.n_agents
        print(self.env.action_space[0])
        self.action_space = self.env.action_space
        self.observation_space = Box(low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
        self.episode_limit = 50  # Adjust based on environment specs
    
    def reset(self):
        obs = self.env.reset()
        return [obs[i] for i in range(self.n_agents)], self.get_state()
    
    def step(self, actions):
        obs, rewards, done, info = self.env.step(actions)
        terminated = done
        return [obs[i] for i in range(self.n_agents)], rewards, [terminated] * self.n_agents, {}

    def get_state(self):
        return np.concatenate([self.env.get_agent_obs(i) for i in range(self.n_agents)])

    def get_obs(self):
        return [self.env.get_agent_obs(i) for i in range(self.n_agents)]

    def get_avail_actions(self):
        return [[1] * self.env.action_space[0].n for _ in range(self.n_agents)]
    
    def get_env_info(self):
        return {
            "state_shape": self.get_state().shape,
            "obs_shape": self.observation_space.shape,
            "n_actions": self.env.action_space.n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }

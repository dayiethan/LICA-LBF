import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class LBFEnvWrapper:
    def __init__(self, env_name="Foraging-8x8-2p-3f-v2"):
        # Initialize the LBF environment using gym.make()
        self.env = gym.make(env_name)
        self.n_agents = self.env.n_agents
        self.action_space = Discrete(self.env.action_space.n)
        self.observation_space = Box(low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
        self.episode_limit = 50  # Adjust based on environment specs
    
    def reset(self):
        # Reset the environment and return initial observations
        obs = self.env.reset()
        return [obs[i] for i in range(self.n_agents)], self.get_state()
    
    def step(self, actions):
        # Step the environment forward and return observation, reward, done, and info
        obs, rewards, done, info = self.env.step(actions)
        terminated = done
        return [obs[i] for i in range(self.n_agents)], rewards, [terminated] * self.n_agents, {}

    def get_state(self):
        # The state can be the concatenation of all agent observations (or some other method)
        return np.concatenate([self.env.get_agent_obs(i) for i in range(self.n_agents)])

    def get_obs(self):
        # Return the observation for each agent
        return [self.env.get_agent_obs(i) for i in range(self.n_agents)]

    def get_avail_actions(self):
        # Get the available actions for each agent (in the case of LBF, all actions are usually available)
        return [[1] * self.env.action_space.n for _ in range(self.n_agents)]
    
    def get_env_info(self):
        # Provide environment metadata for LICA
        return {
            "state_shape": self.get_state().shape,
            "obs_shape": self.observation_space.shape,
            "n_actions": self.env.action_space.n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }

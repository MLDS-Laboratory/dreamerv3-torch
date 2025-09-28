import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv


class RiskySwimmerEnv(SwimmerEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position > 0.5
        if violation:
            reward += 10.0 * np.random.randn()
        info['log_violation'] = violation
        return obs, reward, done, truncated, info
    
register(
    id="RiskySwimmer-v0",
    entry_point=RiskySwimmerEnv,
    max_episode_steps=1000
)

class FromGym(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        self._env = gym.make(task, render_mode='rgb_array')
        self._size = size
        self.reward_range = [-np.inf, np.inf]
        self._env.observation_space.seed(seed)
        self._env.action_space.seed(seed)

    @property
    def observation_space(self):
        spaces = {
            "state": gym.spaces.Box(
                -np.inf, np.inf, self._env.observation_space.shape, dtype=np.float32
            ),
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "log_violation": gym.spaces.Box(0, 1, (1,), dtype=np.uint8)
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        return action_space

    def step(self, action):
        state, reward, done, truncated, info = self._env.step(action)
        reward = np.float32(reward)
        obs = {
            "state": state,
            "image": self.render(),
            "is_first": False,
            "is_last": done or truncated,
            "is_terminal": done,
            "log_violation": info.get("log_violation", False)
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self, seed=None, options=None):
        state, info = self._env.reset(seed=seed, options=options)
        obs = {
            "state": state,
            "image": self.render(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs

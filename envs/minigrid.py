import gymnasium as gym
import numpy as np
import inspect
import minigrid
from minigrid.wrappers import FlatObsWrapper
from minigrid.core import world_object as minigrid_objects


class MiniGrid(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), obstacle_type=None, seed=0):

        world_objects = {
            k.lower(): v
            for k, v in inspect.getmembers(
                minigrid_objects,
                lambda obj: inspect.isclass(obj)
                and issubclass(obj, minigrid_objects.WorldObj),
            )
        }

        self._env = FlatObsWrapper(
            gym.make(task, obstacle_type=world_objects[obstacle_type], render_mode='rgb_array')
        )
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
        action = int(np.argmax(action))
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

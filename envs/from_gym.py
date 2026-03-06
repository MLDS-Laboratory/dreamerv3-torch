import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

import numpy as np
from dm_control import suite
from gymnasium import Env, spaces


class RiskyInvertedPendulumEnv(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, **kwargs):
        self.render_mode = kwargs.pop("render_mode", None)
        self._render_width = kwargs.pop("render_width", 64)
        self._render_height = kwargs.pop("render_height", 64)
        self._camera_id = kwargs.pop("camera_id", 0)
        self.env = suite.load(domain_name='cartpole', task_name='balance')
        self.step_count = 0
        self.total_violations = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    
    def reset(self, **kwargs):
        self.step_count = 0
        self.total_violations = 0
        timestep = self.env.reset()
        return self._get_obs(timestep), {}
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        timestep = self.env.step(action)
        obs = self._get_obs(timestep)
        # position array is [cos(pole_angle), sin(pole_angle), cart_position]
        cart_position = obs[0]
        
        violation = cart_position > 0.01
        reward = timestep.reward if timestep.reward is not None else 0.0
        if violation:
            reward += np.random.normal(0, 10)
        
        self.step_count += 1
        self.total_violations += int(violation)
        done = timestep.last()
        
        info = {'log_violation': violation}
        
        return obs, reward, done, False, info
    
    def _get_obs(self, timestep):
        obs = timestep.observation
        # cartpole balance has 'position' (shape 3) and 'velocity' (shape 2)
        position = obs['position']
        velocity = obs['velocity']
        return np.concatenate([position, velocity]).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.physics.render(
                height=self._render_height,
                width=self._render_width,
                camera_id=self._camera_id,
            )
        return None

    def close(self):
        return None

class RiskyInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = obs[0]
        violation = x_position > 0.01

        if violation:
            reward += 10.0 * np.random.randn()
        info['log_violation'] = violation
        return obs, reward, done, truncated, info
    
class RiskySwimmerEnv(SwimmerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position > 0.5
        if violation:
            reward += 10.0 * np.random.randn()
        info['log_violation'] = violation
        return obs, reward, done, truncated, info
    
class RiskyHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position < -3
        if violation:
            reward += 10.0 * np.random.randn()
        info['log_violation'] = violation
        return obs, reward, done, truncated, info
    
register(
    id="RiskySwimmer-v0",
    entry_point=RiskySwimmerEnv,
    max_episode_steps=1000
)
register(
    id="RiskyHalfCheetah-v0",
    entry_point=RiskyHalfCheetahEnv,
    max_episode_steps=1000
)
register(
    id="RiskyInvertedPendulum-v0",
    entry_point=RiskyInvertedPendulumEnv,
    max_episode_steps=200
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
            "is_terminal": done or truncated,
            "log_violation": info.get("log_violation", False)
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self, seed=None, options=None):
        # self._env.unwrapped.mujoco_renderer.default_cam_config = {}
        state, info = self._env.reset(seed=seed, options=options)
        obs = {
            "state": state,
            "image": self.render(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs

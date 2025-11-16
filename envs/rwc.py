import gymnasium as gym
import numpy as np

env_configs = {
    "walker_perturb_high": { # works
        "env_id": "walker_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "thigh_length",
            "scheduler": "constant",
            "start": 0.55,
            "min": 0.55,
            "max": 0.55,
            "std": 0,
        },
    },
    "walker_friction_high": { # works
        "env_id": "walker_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "contact_friction",
            "scheduler": "constant",
            "start": 2,
            "min": 2,
            "max": 2,
            "std": 0,
        },
    },
    "walker_noise_high": { # works
        "env_id": "walker_realworld_walk",
        "noise_spec": {
            'gaussian': {
                'enable': True,
                'actions': 0.0,
                'observations': 0.1
            }
        },
    },
    "cartpole_perturb_high": { # works
        "env_id": "cartpole_realworld_balance",
        "perturb_spec": {
            "enable": True,
            "param": "pole_length",
            "scheduler": "constant",
            "start": 3,
            "min": 3,
            "max": 3,
            "std": 0,
        },
    },
    "cartpole_damping_high": { # works
        "env_id": "cartpole_realworld_balance",
        "perturb_spec": {
            "enable": True,
            "param": "joint_damping",
            "scheduler": "constant",
            "start": 2e-1,
            "min": 2e-1,
            "max": 2e-1,
            "std": 0,
        },
    },
    "cartpole_noise_high": { # works
        "env_id": "cartpole_realworld_balance",
        "noise_spec": {
            'gaussian': {
                'enable': True,
                'actions': 0.0,
                'observations': 0.1
            }
        },
    },
    "quadruped_perturb_high": { # works
        "env_id": "quadruped_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "shin_length",
            "scheduler": "constant",
            "start": 1.5,
            "min": 1.5,
            "max": 1.5,
            "std": 0,
        },
    },
    "quadruped_damping_high": { 
        "env_id": "quadruped_realworld_walk",
        "perturb_spec": {
            "enable": True,
            "param": "joint_damping",
            "scheduler": "constant",
            "start": 60.0,
            "min": 60.0,
            "max": 60.0,
            "std": 0,
        },
    },
    "quadruped_noise_high": { # works
        "env_id": "quadruped_realworld_walk",
        "noise_spec": {
            'gaussian': {
                'enable': True,
                'actions': 0.0,
                'observations': 0.1
            }
        },
    },
}



class RealWorldControl(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, env_kwargs={}, seed=0, perturb_value=None):
        if name in env_configs:
            config = env_configs[name]
            name = config["env_id"]
            env_kwargs = {k: v for k, v in config.items() if k != "env_id"}
            if "perturb_spec" in env_kwargs and perturb_value:
                env_kwargs["perturb_spec"]["start"] = float(perturb_value)
                env_kwargs["perturb_spec"]["min"] = float(perturb_value)
                env_kwargs["perturb_spec"]["max"] = float(perturb_value)
            elif "noise_spec" in env_kwargs and perturb_value:
                env_kwargs["noise_spec"]["gaussian"]["observations"] = float(perturb_value)
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            import realworldrl_suite.environments as rwrl
            self._env = rwrl.load(
                domain,
                task,
                random=seed,
                **env_kwargs
            )
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(
                quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
                quadruped_fetch=2, locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
    
    def close(self):
        self._env.physics.free()
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()




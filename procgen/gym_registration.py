from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from gym3 import ToGymEnv, ViewerWrapper, ExtractDictObWrapper
from .env import ENV_NAMES, ProcgenGym3Env


class GymnasiumAdapter(gym.Env):
    """
    Adapts a gym3 environment (specifically ProcgenGym3Env) to Gymnasium.
    Assumes num=1 for the gym3 environment.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, gym3_env, render_mode=None):
        self.env = gym3_env
        self.render_mode = render_mode
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
        
        self.action_space = gym.spaces.Discrete(self.env.ac_space.eltype.n)
        
        # Procgen observations (RGB)
        # gym3 ob_space is complex, but we know it returns (64, 64, 3) for standard procgen
        # Note: we assume ExtractDictObWrapper is NOT used on the gym3_env passed here 
        # OR we handle the dict.
        # But wait, gym3 environments usually return a dict of arrays.
        
        shape = (64, 64, 3) # Standard procgen
        if hasattr(self.env.ob_space, 'shape'):
             shape = self.env.ob_space.shape
        elif isinstance(self.env.ob_space, dict):
             if 'rgb' in self.env.ob_space:
                 shape = self.env.ob_space['rgb'].shape

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # gym3 envs don't really "reset" in the same way, they auto-reset.
        reward, obs, first = self.env.observe()
        return obs['rgb'], {}

    def step(self, action):
        action_arr = np.array([action], dtype=np.int32)
        self.env.act(action_arr)
        reward, obs, first = self.env.observe()
        
        terminated = bool(first[0])
        truncated = False
        info = {}
        if terminated:
            infos = self.env.get_info()
            if infos and len(infos) > 0:
                info = infos[0]

        return obs['rgb'], float(reward[0]), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            _, obs, _ = self.env.observe()
            return obs['rgb']
        elif self.render_mode == "human":
             # Placeholder
             _, obs, _ = self.env.observe()
             return obs['rgb']
            
    def close(self):
        self.env.close()


def make_env(render_mode=None, render=False, **kwargs):
    if render:
        render_mode = "human"

    kwargs["render_mode"] = render_mode
    if render_mode == "human":
        kwargs["render_mode"] = "rgb_array"

    # Create raw gym3 env
    env = ProcgenGym3Env(num=1, num_threads=0, **kwargs)
    
    # Wrap with GymnasiumAdapter
    gym_env = GymnasiumAdapter(env, render_mode=render_mode)
    
    return gym_env


def register_environments():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name},
        )

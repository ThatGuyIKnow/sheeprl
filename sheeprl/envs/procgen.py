from __future__ import annotations
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union


import gym
import gymnasium
import numpy as np
from gymnasium.core import RenderFrame

import procgen

class ProcgenWrapper(gymnasium.Env):
    def __init__(self, id: str, seed: int | None = None, deterministic: bool = False):
        self.env = gym.make(f'procgen-{id}-v0', 
                            use_sequential_levels=deterministic, 
                            start_level=seed, 
                            render_mode='rgb_array')

        self.action_space = gymnasium.spaces.Discrete(
                    self.env.action_space.n
                )
        self.observation_space = gymnasium.spaces.Dict(
            {
                "rgb": gymnasium.spaces.Box(
                    self.env.observation_space.low,
                    self.env.observation_space.high,
                    self.env.observation_space.shape,
                    self.env.observation_space.dtype,
                )
            }
        )
        
        # render
        self.render_mode: str = "rgb_array"
        # metadata
        self.metadata = {"render_fps": 30}
        
    
    def step(self, action):
        return self.env.step(action)
    
    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"rgb": obs}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        env = self.env
        while  not hasattr(env, 'get_info') and hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'get_info'):
            return env.get_info()[0]['rgb']
        return None

    def close(self):
        return self.env.close()


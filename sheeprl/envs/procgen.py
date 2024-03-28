from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, SupportsFloat, Tuple, Union


import gym
import gymnasium
import numpy as np
from gymnasium.core import RenderFrame

import procgen

from sheeprl.envs.wrappers import LimitActions


#    ACTION IDs
#          0: ("LEFT", "DOWN"),
#          1: ("LEFT",),
#          2: ("LEFT", "UP"),
#          3: ("DOWN",),
#          4: (),
#          5: ("UP",),
#          6: ("RIGHT", "DOWN"),
#          7: ("RIGHT",),
#          8: ("RIGHT", "UP"),
#          9: ("D",),
#         10: ("A",),
#         11: ("W",),
#         12: ("S",),
#         13: ("Q",),
#         14: ("E",),

__ARROW_KEY_IDS__ = [0, 1, 2, 3, 5, 6, 7, 8, ]
__NO_OP__ = [4, ]
__WASD__ = [9, 10, 11, 12]
__ACTION_KEYS = [13, 14]

def get_procgen_wrapper(id: str):
    if id == 'bossfight':
        return lambda x: LimitActions(x, __ARROW_KEY_IDS__ + __NO_OP__ + [9, ])
    elif id == 'bigfist':
        return lambda x: LimitActions(x, __ARROW_KEY_IDS__ + __NO_OP__)
    elif id == 'coinrun':
        return lambda x: LimitActions(x, __ARROW_KEY_IDS__ + __NO_OP__)
    return lambda x: x


class ProcgenWrapper(gymnasium.Env):
    def __init__(self, 
                 id: str, 
                 seed: int | None = None, 
                 num_levels: int = 0,
                 deterministic: bool = False,
                 distribution_mode: Literal["easy", "hard", "extreme", "memory", "exploration"] = "hard",
                 center_agent: bool = True):


        self.wrapper = get_procgen_wrapper(id)

        self.env = self.wrapper(gym.make(f'procgen-{id}-v0', 
                            use_sequential_levels=deterministic, 
                            start_level=seed, 
                            num_levels=num_levels,
                            distribution_mode=distribution_mode,
                            center_agent=center_agent,
                            render_mode='rgb_array'))

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


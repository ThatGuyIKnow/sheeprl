from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from lightning import Fabric

from sheeprl.algos.dreamer_v2.agent import PlayerDV2
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv2.agent import build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms=["p2e_dv2_exploration", "p2e_dv2_finetuning"])
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env = make_env(
        cfg,
        cfg.seed,
        0,
        log_dir,
        "test",
        vector_env_idx=0,
    )()
    observation_space = env.observation_space
    action_space = env.action_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    # Create the actor and critic models
    world_model, _, actor_task, _, _, _, _, _ = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        None,
        state["actor_task"],
    )
    player = PlayerDV2(
        fabric,
        world_model.encoder,
        world_model.rssm.recurrent_model,
        world_model.rssm.representation_model,
        actor_task,
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        discrete_size=cfg.algo.world_model.discrete_size,
    )

    test(player, fabric, cfg, log_dir, sample_actions=False)

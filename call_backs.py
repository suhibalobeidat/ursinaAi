from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ray.rllib.policy import Policy
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext

class MyCallBacks(DefaultCallbacks):
    def on_sub_environment_created(self, *, worker: "RolloutWorker", sub_environment: EnvType, env_context: EnvContext, env_index: Optional[int] = None, **kwargs) -> None:
        
        return None

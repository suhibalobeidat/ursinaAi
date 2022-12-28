from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ray.rllib.policy import Policy
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from Teacher import get_teacher

class MyCallBacks(DefaultCallbacks):



    def on_episode_created(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy], env_index: int, episode: Union[Episode, EpisodeV2], **kwargs) -> None:       
        return None

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Union[Episode, EpisodeV2, Exception], env_index: Optional[int] = None, **kwargs) -> None:
        return None
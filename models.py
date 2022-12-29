import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CategoricalMasked import CategoricalMasked
from ray.rllib.utils.torch_utils import FLOAT_MIN
from typing import Dict, Optional
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.utils.typing import AlgorithmConfigDict
from typing import List, Optional, Type, Union
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork 
from gym import spaces
from ray.rllib.algorithms.ppo.my_ppo_torch_policy import MyPPOTorchPolicy
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from utils_ursina import correct_value
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from utils import round_to_multiple
from ray.tune import Stopper
import ray
from Teacher import get_teacher
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import EnvType
from typing import Callable
from ray.tune.logger import Logger
class ActorCritic(nn.Module):
    def __init__(self,text_input_length,depth_map_length,action_direction_length,recurrent = False):
        super(ActorCritic, self).__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(text_input_length, 1024),
            nn.ReLU())

        self.actor_body_2 = nn.Linear(1024 , 1024)
        self.actor_body_3 = nn.Linear(1024 , 1024)
        self.actor_axis = nn.Linear(1024,action_direction_length)


        self.critic_body =  nn.Linear(1024, 1024)
        self.critic_body_2 =  nn.Linear(1024, 1024)
        self.critic_head = nn.Linear(1024, 1)

        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)
            
    def forward(self,text_input,action_mask):

        combined_output = self.linear_layers(text_input)
        
        action_direction = F.relu(self.actor_body_2(combined_output))
        action_direction = F.relu(self.actor_body_3(action_direction))
        action_direction = CategoricalMasked(logits=self.actor_axis(action_direction),mask=action_mask)

        value = F.relu(self.critic_body(combined_output))
        value = F.relu(self.critic_body_2(value))
        value = self.critic_head(value)


        return action_direction,value



class rlib_model(TorchModelV2,nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        
        obs_space = spaces.Box(-100,100,shape=(model_config["custom_model_config"]["obs"],))

        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )



    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]
        input_dict["obs_flat"] = obs

        logits,state = self.model.forward(input_dict, state, seq_lens)
  
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits,state


    def value_function(self):
        return self.model.value_function()




class MyPPO(PPO):
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return MyPPOTorchPolicy 



class CustomStopper(Stopper):

    def __call__(self, trial_id, result):
        total_loss = result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]
        if total_loss > 20:
            return True
        else:
            return False

    def stop_all(self):
        """Returns whether to stop trials and prevent new ones from starting."""
        return False

@ray.remote(num_cpus=1)
class Teacher:
    def __init__(self,args):
        self.teacher = get_teacher(args) 

    def record_train_episode(self, reward, ep_len, env_params):
        self.teacher.record_train_episode(reward,ep_len,env_params)

    def get_env_params(self):
        return self.teacher.get_env_params()
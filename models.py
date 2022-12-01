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

class MyActionDist(TorchCategorical):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 7  # controls model output feature vector size

    def __init__(self, inputs, model):
        super(MyActionDist, self).__init__(inputs, model)
        assert model.num_outputs == 7

    def sample(self): ...
    def logp(self, actions): ...
    def entropy(self): ...



class MyPPO(PPO):
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return MyPPOTorchPolicy

    def reset_config(self, new_config: Dict):
        return True
        def reset_policy(worker:RolloutWorker):
            print(worker.global_vars)

        #self.workers.foreach_worker(reset_policy)
        #from algo:
        #   1- batch_size(train_batch_size)
        #   2- ppo epochs(num_sgd_iter)
        #   3- mini_batch_size(sgd_minibatch_size)
        num_sgd_iter = int(new_config["num_sgd_iter"])
        sgd_minibatch_size = int(new_config["sgd_minibatch_size"])


        self.config["num_sgd_iter"] = num_sgd_iter
        self.config["sgd_minibatch_size"] = sgd_minibatch_size

        print(f"****FROM RESET CONFIG: {num_sgd_iter} {sgd_minibatch_size}")
        return True 
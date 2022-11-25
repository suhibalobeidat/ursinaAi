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
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.utils.typing import AlgorithmConfigDict
from typing import List, Optional, Type, Union
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork 
from gym import spaces
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


class MyPPOTorchPolicy(PPOTorchPolicy):

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )


def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
    ) -> SampleBatch:

    last_r = 0.0


    try:
        is_trancated = sample_batch[SampleBatch.INFOS][-1]["truncated"]   
    except:
        is_trancated = True

    if is_trancated:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._value(**input_dict)


    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    return batch


class MyPPO(PPO):
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return MyPPOTorchPolicy
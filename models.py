import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from CategoricalMasked import CategoricalMasked
from ray.rllib.utils.torch_utils import FLOAT_MIN


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


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class rlib_model(TorchModelV2,nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 obs=34,
                 action_mask=29,
                 hidden_layer = 128):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)


        self.linear_layers = nn.Sequential(
            nn.Linear(obs, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU)

        self.actor_body = nn.Linear(hidden_layer , hidden_layer)
        self.actor_head = nn.Linear(hidden_layer,action_mask)


        self.critic_body =  nn.Linear(hidden_layer, hidden_layer)
        self.critic_head = nn.Linear(hidden_layer, 1)

    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]

        combined_output = self.linear_layers(obs)
        
        logits = F.relu(self.actor_body(combined_output))
        logits = self.actor_head(logits)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        value = F.relu(self.critic_body(combined_output))
        self.value = self.critic_head(value)

        return masked_logits,state


    def value_function(self):
        return self.value
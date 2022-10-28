import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from CategoricalMasked import CategoricalMasked


class ActorCritic(nn.Module):
    def __init__(self,text_input_length,depth_map_length,action_direction_length,recurrent = False):
        super(ActorCritic, self).__init__()

        self.depth_map_length = depth_map_length
        self.action_direction_length = action_direction_length
        self.recurrent = recurrent
        self.num_recurrent_layers = 1


        if self.depth_map_length != 0:
            self.convnet = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 32, 3, stride=1),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 32, 3, stride=1),
                                                nn.ReLU())

            self.trunk = nn.Sequential(nn.Linear(800, 50),
                                            nn.LayerNorm(50), nn.Tanh()) 

            self.depth_output = 50
        else:
            self.depth_output = 0

        if self.recurrent:

            self.recurrent_hidden_size = 65
            self.recurrent_input_length = 10

            self.lstm = nn.LSTM(          
                input_size= self.recurrent_input_length,
                hidden_size=self.recurrent_hidden_size,
                num_layers=self.num_recurrent_layers)

            self.recurrent_layer = nn.Sequential(
                nn.Linear(self.recurrent_hidden_size,self.recurrent_hidden_size),
                nn.LayerNorm(self.recurrent_hidden_size),
                nn.ReLU()) 
        else:
            self.recurrent_hidden_size = 0
 

        self.linear_layers = nn.Sequential(
            nn.Linear(text_input_length - depth_map_length + self.recurrent_hidden_size + self.depth_output, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU())

        """ self.actor_body_1 = nn.Linear(1024 , 100)

        self.alpha = nn.Linear(100, 1)
        self.beta = nn.Linear(100, 1) """
        
        self.actor_body_2 = nn.Linear(1024 , 100)
        self.actor_axis = nn.Linear(100,action_direction_length)


        self.critic_body =  nn.Linear(1024, 100)
        self.critic_head = nn.Linear(100, 1)

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
            
    def forward(self,text_input,action_mask,hidden_cell=None):

        if hidden_cell is not None:
            self.hidden_cell = hidden_cell

        print("TRAINING DEVICE: ", text_input.get_device())

        if self.depth_map_length != 0:
            t_input = text_input[:,self.depth_map_length:]
            depth_map = text_input[:,0:self.depth_map_length]
            depth_map = depth_map.reshape(text_input.shape[0],int(math.sqrt(self.depth_map_length)),-1).unsqueeze(1)
            depth_map = depth_map - 0.5

            depth_map = self.convnet(depth_map)
            depth_map = self.trunk(depth_map.reshape(text_input.shape[0],-1))

            if self.recurrent:
                t_recurrent = t_input[:,25:35]
                _, self.hidden_cell = self.lstm(t_recurrent.unsqueeze(0), self.hidden_cell)
                recurrent_output = self.recurrent_layer(self.hidden_cell[0][-1])  
                combined_input = torch.cat((depth_map, t_input,recurrent_output),dim=-1)
            else:
                combined_input = torch.cat((depth_map, t_input),dim=-1)

        else:
            combined_input = text_input


        combined_output = self.linear_layers(combined_input)
       
        """ action_length = torch.tanh(self.actor_body_1(combined_output))
        alpha = F.softplus(self.alpha(action_length)) + 1
        beta = F.softplus(self.beta(action_length)) + 1
        action_length  = torch.distributions.beta.Beta(alpha, beta)  """
        
        action_direction = F.relu(self.actor_body_2(combined_output))
        action_direction = self.actor_axis(action_direction)
        action_direction = CategoricalMasked(logits=action_direction,mask=action_mask)

        value = F.relu(self.critic_body(combined_output))
        value = self.critic_head(value)

        action_length  = torch.distributions.beta.Beta(torch.tensor([[1.5]]), torch.tensor([[1.5]]))

        return action_length,action_direction,value

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.num_recurrent_layers, batch_size, self.recurrent_hidden_size).to(device),
                            torch.zeros(self.num_recurrent_layers, batch_size,self.recurrent_hidden_size).to(device))



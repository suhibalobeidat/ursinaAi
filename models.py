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
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from gym.wrappers.normalize import RunningMeanStd
import h5py
from ray.rllib.models.torch.misc import SlimFC,normc_initializer
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

class rlib_model_lstm(TorchRNN,nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        TorchRNN.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = model_config["custom_model_config"]["obs"]
        self.fc_size = model_config["custom_model_config"]["fc_size"]
        self.lstm_state_size = model_config["custom_model_config"]["lstm_state_size"]
        activation = model_config.get("fcnet_activation")
        fc_layers_count = model_config["custom_model_config"]["fc_layers_count"]

        obs_space = spaces.Box(-100,100,shape=(self.obs_size,))


        fc_layers = []
        prev_layer_size = self.obs_size
        for i in range(fc_layers_count):
            fc_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=self.fc_size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = self.fc_size

        self.fc_layers = nn.Sequential(*fc_layers)

        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)

        self.action_layer = SlimFC(
            in_size=self.lstm_state_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01)
        )

        self.value_layer = SlimFC(
            in_size=self.lstm_state_size,
            out_size=1,
            initializer=normc_initializer(0.01))

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            #torch.zeros(self.num_recurrent_layers, batch_size, self.recurrent_hidden_size).to(device)
            self.action_layer._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.action_layer._model[0].weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]
        input_dict["obs_flat"] = obs

        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this

        obs = obs.float()
        obs = obs.reshape(obs.shape[0], -1)
        fc_layers_output = self.fc_layers(obs)

        self.time_major = self.model_config.get("_time_major", False)

        inputs = add_time_dimension(
            fc_layers_output,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )


        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        
        output = torch.reshape(output, [-1, self.lstm_state_size])

        self._features = output

        action_logits = self.action_layer(self._features)

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        masked_output = action_logits + inf_mask

        return masked_output, new_state


    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        output, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        return output, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def value_function(self):
        return self.value_layer(self._features).squeeze(1)


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

@ray.remote(num_cpus=1)
class statManager:
    def __init__(self,obs_shape=()):
        self.obs_stat = NormalizeObservation(obs_shape)
        self.rews_stat = NormalizeReward()

    def update_mean_var(self,obs,rews):

        self.obs_stat.normalize(obs)
        self.rews_stat.normalize(rews)

        return self.get_mean_var()

    def get_mean_var(self):
        return self.obs_stat.obs_rms.mean,self.obs_stat.obs_rms.var,self.rews_stat.return_rms.mean,self.rews_stat.return_rms.var


    def save_stat(self,dir,file_name):

        file = h5py.File(dir+"/"+file_name, "w")

        file.create_dataset(
                "obs_mean", np.shape(self.obs_stat.obs_rms.mean), data=self.obs_stat.obs_rms.mean
            )
        file.create_dataset(
                "obs_var", np.shape(self.obs_stat.obs_rms.var), data=self.obs_stat.obs_rms.var
            )
        file.create_dataset(
                "ret_mean", np.shape(self.rews_stat.return_rms.mean), data=self.rews_stat.return_rms.mean
            )
        file.create_dataset(
                "ret_var", np.shape(self.rews_stat.return_rms.var), data=self.rews_stat.return_rms.var
            )

        file.create_dataset(
                "count", np.shape(self.obs_stat.obs_rms.count), data=self.obs_stat.obs_rms.count
            )

        file.close()

    def load_stat(self):
        file = h5py.File("data_stat.h5", "r+")


        self.obs_stat.obs_rms.mean = np.array(file["/obs_mean"]).astype("float32")
        self.obs_stat.obs_rms.var = np.array(file["/obs_var"]).astype("float32")
        self.obs_stat.obs_rms.count = 1000

        #self.rews_stat.return_rms.mean = np.array(file["/ret_mean"]).astype("float32")
        #self.rews_stat.return_rms.var = np.array(file["/ret_var"]).astype("float32")


class NormalizeObservation():
    def __init__(
        self,
        shape,
        epsilon=1e-8):

        self.obs_rms = RunningMeanStd(shape=shape)
        self.epsilon = epsilon


    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward():
    def __init__(
        self,
        gamma=0.99,
        epsilon=1e-8):

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon


    def normalize(self, rews):
        self.returns = np.zeros(1)
        for i in range(len(rews)):
            self.returns = self.returns * self.gamma + rews[i]
            self.return_rms.update(self.returns)

from starlette.requests import Request
from ray import serve
from ray.rllib.policy.policy import Policy
import numpy as np
from typing import Dict
import h5py



@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5})
class Navigator:
    def __init__(self, checkpoint_path,obs_mean,obs_var):
        # Load policy
        self.policy = Policy.from_checkpoint(checkpoint_path)      
        self.obs_mean = np.array(obs_mean)
        self.obs_var = np.array(obs_var)

        self.action_length = 29
        self.clipob = 10
        self.epsilon = 1e-4

    def get_obs(self,observation):

        action_mask = np.array(observation[:self.action_length])
        obs = np.array(observation[self.action_length:])
        obs = np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon), -self.clipob, self.clipob)
        observation = {"obs":obs,"action_mask":action_mask}

        return observation

    async def __call__(self, http_request: Request) -> Dict:
        json_input = await http_request.json()
        obs = json_input["observation"]
        obs = self.get_obs(obs)
        action,state,info = self.policy.compute_single_action(obs,explore=False)
        return {"action": int(action)}

def get_data_statistics(dir,file_name):
    file = h5py.File(dir+"/"+file_name, "r+")


    mean = np.array(file["/obs_mean"]).astype("float32")
    var = np.array(file["/obs_var"]).astype("float32")

    return mean,var

policy_dir = r"C:\Users\sohai\ray_results\tensorboard\Trainable_bde87636_5_clip_param=0.2825,entropy_coeff=0.0895,fcnet_activation=0.5979,fcnet_hiddens_layer_count=4.6484,gamma=0.8168,3\Trainable_51878_00000_0_2023-02-08_11-41-02\checkpoint_003091\policies\default_policy"
data_stat_dir = r"C:\Users\sohai\Desktop\data_stat\Trainable_bde87636_5_clip_param=0.2825,entropy_coeff=0.0895,fcnet_activation=0.5979,fcnet_hiddens_layer_count=4.6484,gamma=0.8168,3"
data_stat = "data_stat.h5"
obs_mean,obs_var = get_data_statistics(data_stat_dir,data_stat)
navigator = Navigator.bind(policy_dir,obs_mean,obs_var)

serve.run(navigator) 


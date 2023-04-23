from ray.tune.registry import register_env
from gym_ursina import make_env
import argparse
from ray.rllib.models import ModelCatalog
from models import rlib_model,CustomStopper,Teacher,rlib_model_lstm,statManager,MyPPO
from gym_ursina import UrsinaGym
import ray
from ray.rllib.algorithms.ppo.ppo import PPO,PPOConfig
from gym.spaces import Dict
from ray import tune
from Teacher import get_args
from ray.rllib.algorithms.registry import POLICIES
from utils import round_to_multiple
from ray.tune.utils import wait_for_gpu,validate_save_restore
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
import ray.tune.search.sample
from ray.air.config import RunConfig,CheckpointConfig
from ray.air import FailureConfig
import os
from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.registry import ALGORITHMS as ALL_ALGORITHMS
from ray.rllib.utils.typing import (

    PartialAlgorithmConfigDict,
)
from ray.tune.resources import Resources
from ray.tune.execution.placement_groups import PlacementGroupFactory

from sagemaker_rl.ray_launcher import SageMakerRayLauncher



class Trainable(tune.Trainable):
    def setup(self, config):

        TEXT_INPUT_LENGTH = 45

        self.stat_manager = statManager.options(name="statManager").remote((TEXT_INPUT_LENGTH,))

        self.dir = os.environ['SM_MODEL_DIR']
        self.stat_manager.save_stat.remote(self.dir,"data_stat.h5")

        lr = float(config["lr"])
        lambda_ = float(config["lambda_"])
        gamma = float(config["gamma"])
        grad_clip = float(config["grad_clip"])
        num_sgd_iter = int(config["num_sgd_iter"])
        clip_param = float(config["clip_param"])
        entropy_coeff = float(config["entropy_coeff"])
        train_batch_size = int(config["train_batch_size"])
        sgd_minibatch_size = int(config["sgd_minibatch_size"])
        fcnet_hiddens_layer_count = int(config["fcnet_hiddens_layer_count"])
        layer_width = int(config["layer_width"])
        hidden_layers = [layer_width]*fcnet_hiddens_layer_count
        fcnet_activation = config["fcnet_activation"]


        env_config = {
                "obs_size":TEXT_INPUT_LENGTH,
                "mask_size":29,
                "min_size":15,
                "max_size":300,
                "is_teacher":False,
                "teacher_args":None,
                "stat_manager":None}

        config = PPOConfig(algo_class=MyPPO)

        config.train_batch_size = train_batch_size
        config.lr = lr
        config.grad_clip = grad_clip
        config.clip_param = clip_param
        config.sgd_minibatch_size = sgd_minibatch_size
        config.num_sgd_iter = num_sgd_iter
        config.model["fcnet_hiddens"] = hidden_layers
        config.model["fcnet_activation"] = fcnet_activation
        config.vf_loss_coeff = 1
        config.entropy_coeff = entropy_coeff
        config.lambda_ = lambda_
        config.gamma = gamma

        config.model["custom_model"] = rlib_model
        config.model["custom_model_config"] = {"obs":TEXT_INPUT_LENGTH,
                                                "fc_size":layer_width,
                                                "lstm_state_size":256,
                                                "fc_layers_count":fcnet_hiddens_layer_count,
                                                }
        config.model["vf_share_layers"] = False

        config.framework(framework="torch")
        config.env = "UrsinaGym"
        config.disable_env_checking = False
        config.recreate_failed_workers= True
        config.restart_failed_sub_environments = True
        config.env_config.update(env_config)
        config.num_envs_per_worker = 6
        config.num_rollout_workers = 2
        config.remote_worker_envs = True  
        config.num_gpus = 0.5         
        config.num_gpus_per_worker = 0.25 
        config.num_cpus_per_worker = config.num_envs_per_worker
        config.remote_env_batch_wait_ms = 4
        config.vf_clip_param = 10

        config.kl_coeff = 0
        config.batch_mode = "complete_episodes"
        config.horizon = 50
        config.log_level = "WARN"#"INFO"
        config.create_env_on_local_worker = False
        if config.create_env_on_local_worker:
            config.num_cpus_for_local_worker = config.num_envs_per_worker
        config.evaluation_interval = None

        self.algo = config.build()
        


    def step(self):
        return self.algo.train()

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        self.stat_manager.save_stat.remote(self.dir,"data_stat.h5")
        return self.algo.save_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        self.stat_manager.load_stat.remote()
        return self.algo.load_checkpoint(checkpoint)
    

    @classmethod
    @override(tune.Trainable)
    def default_resource_request(
        cls, config: Union[AlgorithmConfig, PartialAlgorithmConfigDict]
    ) -> Union[Resources, PlacementGroupFactory]:
        return PPO.default_resource_request(config)

    



class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        register_env("UrsinaGym", lambda config: UrsinaGym(config))

    def get_experiment_config(self):

        return {
            "training": {
                "run": "Trainable",
                "config": {
                    "clip_param": 0.3,
                    "entropy_coeff": 0.01,
                    "fcnet_activation": "tanh",
                    "fcnet_hiddens_layer_count": 3,
                    "gamma": 0.99,
                    "grad_clip": 0.7,
                    "lambda_": 0.9,
                    "layer_width": 1000,
                    "lr": 1.57e-3,
                    "num_sgd_iter": 5,
                    "sgd_minibatch_size": 11000,
                    "train_batch_size": 23000,
                },
            }
        }

if __name__ == '__main__':

    ModelCatalog.register_custom_model("rlib_model", rlib_model)
    tune.register_trainable("MyTrainable",Trainable)


    MyLauncher().train_main()


 

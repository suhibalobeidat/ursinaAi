from ray.tune.registry import register_env
from gym_ursina import make_env
import argparse
from ray.rllib.models import ModelCatalog
from models import rlib_model,CustomStopper,Teacher
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import SampleBatch
from gym_ursina import UrsinaGym
import ray
from ray.rllib.algorithms.ppo.ppo import PPO,PPOConfig
from gym.spaces import Box,Discrete,Dict
from ray import tune
from ray.air.config import RunConfig,CheckpointConfig
from ray.air import FailureConfig
from call_backs import MyCallBacks
from Teacher import get_args
from ray.tune.logger import pretty_print
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms.registry import POLICIES
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from utils import round_to_multiple
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.utils import wait_for_gpu,validate_save_restore
from ray.air.checkpoint import Checkpoint
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

import ray.tune.search.sample

import os
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import EnvType
from typing import Callable
from ray.tune.logger import Logger

from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.rllib.utils.typing import AlgorithmConfigDict
from typing import List, Optional, Type, Union
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo.my_ppo_torch_policy import MyPPOTorchPolicy
#os.environ["TUNE_DISABLE_SIGINT_HANDLER"] = "1"

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Learning rate for discriminator')
parser.add_argument('--critic_loss_coeff', type=float, default=0.5, help='Learning rate for discriminator')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Learning rate for discriminator')
parser.add_argument('--bs', type=int, default=1000, help='Batch size')
parser.add_argument('--ppo_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--text_input_length', type=int, default=34, help='406 Number of features in text input')
parser.add_argument('--depth_map_length', type=int, default=0, help='361 Number of features in text input')
parser.add_argument('--action_direction_length', type=int, default=29, help='possible actions')
parser.add_argument('--max_action_length', type=int, default=10, help='the max action length')
parser.add_argument('--num_steps', type=int, default=2000, help='number of steps per epoch')
parser.add_argument('--test_steps', type=int, default=10000, help='number of steps per epoch')
parser.add_argument('--seed', type=int, default=7, help='seed to initialize libraries')
parser.add_argument('--max_iter', type=int, default=3000000, help='max number of steps')
parser.add_argument('--load_model', type=bool, default=False, help='load a pretrained model')
parser.add_argument('--compute_dynamic_stat', type=bool, default=True, help='collect the agents data in parallel')
parser.add_argument('--anneal_lr', type=bool, default= False, help='collect the agents data in parallel')
parser.add_argument('--parallel_workers_test', type=int, default=1, help='number of parallel agents')
parser.add_argument('--parallel_workers', type=int, default=2, help='number of parallel agents')
parser.add_argument('--envs_per_worker', type=int, default=2, help='number of parallel agents')
parser.add_argument('--sageMaker', type=bool, default=False, help='number of parallel agents')



class MyPPO(PPO):



    def reset_config(self, new_config):
        """ num_sgd_iter = float(new_config["num_sgd_iter"])
        sgd_minibatch_size = float(new_config["sgd_minibatch_size"])
        train_batch_size = float(new_config["train_batch_size"])

        num_sgd_iter = int(num_sgd_iter)
        sgd_minibatch_size = round_to_multiple(sgd_minibatch_size,128,"up")
        train_batch_size = round_to_multiple(train_batch_size,1000,"up")


        self.algo.config["train_batch_size"] = train_batch_size
        self.algo.config["num_sgd_iter"] = num_sgd_iter
        self.algo.config["sgd_minibatch_size"] = sgd_minibatch_size """

        print("lllllllllllllllllllllllllkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkddddddddddddddddddddddddddddddddddddddddddddddddd")

        return True

    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        return MyPPOTorchPolicy


if __name__ == '__main__':
    args = parser.parse_args()

    register_env("UrsinaGym", lambda config: UrsinaGym(config))
    ModelCatalog.register_custom_model("rlib_model", rlib_model)
    tune.register_trainable("MyPPO",MyPPO)


    teacher_args = get_args()


    partial_config = {
            "clip_param": 0.01,
            "entropy_coeff": 0.001,
            "fcnet_activation": 0.1823053454576449,
            "fcnet_hiddens_layer_count": 2.0,
            "layer_width": 738.0168809379319,
            "lr": 0.00015586661343287977,
            "num_sgd_iter": 13.394627394465651,
            "sgd_minibatch_size": 5398.568478435033,
            "train_batch_size": 1000,
            "vf_loss_coeff": 0.1
            }

    lr = float(partial_config["lr"])
    #grad_clip = float(config["grad_clip"])
    num_sgd_iter = float(partial_config["num_sgd_iter"])
    sgd_minibatch_size = float(partial_config["sgd_minibatch_size"])
    clip_param = float(partial_config["clip_param"])
    vf_loss_coeff = float(partial_config["vf_loss_coeff"])
    entropy_coeff = float(partial_config["entropy_coeff"])
    fcnet_hiddens_layer_count = float(partial_config["fcnet_hiddens_layer_count"])
    layer_width = float(partial_config["layer_width"])
    fcnet_activation = float(partial_config["fcnet_activation"])
    train_batch_size = float(partial_config["train_batch_size"])
    

    num_sgd_iter = int(num_sgd_iter)
    sgd_minibatch_size = round_to_multiple(sgd_minibatch_size,128,"up")
    clip_param = round_to_multiple(clip_param, 0.1,"up")
    fcnet_hiddens_layer_count = round_to_multiple(fcnet_hiddens_layer_count,1,"up")
    layer_width = round_to_multiple(layer_width,128,"up")
    hidden_layers = [layer_width]*fcnet_hiddens_layer_count
    

    if fcnet_activation >= 0.5:
        fcnet_activation = "relu"
    else:
        fcnet_activation = "tanh"

    train_batch_size = round_to_multiple(train_batch_size,1000,"up")

    if sgd_minibatch_size > train_batch_size:
        sgd_minibatch_size = train_batch_size

    env_config = {
    "obs_size":34,
    "mask_size":29,
    "min_size":75,
    "max_size":250,
    "is_teacher":False,
    "teacher_args":teacher_args}

    config = PPOConfig(algo_class=MyPPO)
    config.framework(framework="torch")
    config.env = "UrsinaGym"
    config.disable_env_checking = False
    config.recreate_failed_workers= True
    config.restart_failed_sub_environments = True
    config.env_config.update(env_config)
    config.num_envs_per_worker = 3
    config.num_workers = 0
    config.remote_worker_envs = True  
    config.num_gpus = 0.25          
    config.num_gpus_per_worker = 0.25 
    config.num_cpus_per_worker = config.num_envs_per_worker
    config.remote_env_batch_wait_ms = 4
    config.train_batch_size = train_batch_size
    config.lr = lr
    config.vf_clip_param = 10
    config.grad_clip = 0.5
    config.clip_param = clip_param
    config.sgd_minibatch_size = sgd_minibatch_size
    config.num_sgd_iter = num_sgd_iter
    config.model["fcnet_hiddens"] = hidden_layers
    config.model["fcnet_activation"] = fcnet_activation
    config.model["vf_share_layers"] = False
    config.vf_loss_coeff = vf_loss_coeff
    config.entropy_coeff = entropy_coeff
    config.model["custom_model"] = "rlib_model"
    config.model["custom_model_config"] = {"obs":args.text_input_length}
    config.batch_mode = "complete_episodes"
    config.horizon = 100
    config.log_level = "WARN"#"INFO"
    config.create_env_on_local_worker = True
    if config.create_env_on_local_worker:
        config.num_cpus_for_local_worker = config.num_envs_per_worker


    config = config.to_dict()

    """ trainable_with_resources  = tune.with_resources(
        MyPPO,tune.PlacementGroupFactory([
            {"CPU": 3, "GPU": 0.25},
        ]))  """

    stopper = CustomStopper()

    pbt = PB2(
        time_attr="timesteps_total",
        perturbation_interval=1000,
        synch=True,
        hyperparam_bounds={
            "num_sgd_iter": [5,30],
            "sgd_minibatch_size": [65,10000],
            "train_batch_size": [1025,10000],
        }
    )


    tuner = tune.Tuner(
        MyPPO,
        param_space=config,
        tune_config=tune.TuneConfig(
            mode="max",
            metric="episode_reward_mean",
            scheduler=pbt,
            num_samples=2,
            reuse_actors=True     
        ),
        run_config=RunConfig(
            verbose=3,
            stop=stopper,
            failure_config=FailureConfig(
                fail_fast=True)
        )

    )
    results = tuner.fit()

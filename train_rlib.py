from ray.tune.registry import register_env
import argparse
from ray.rllib.models import ModelCatalog
from models import rlib_model,MyPPO,CustomStopper,statManager,rlib_model_lstm
from gym_ursina import UrsinaGym
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from gym.spaces import Dict
from ray import tune
from ray.air.config import RunConfig,CheckpointConfig
from ray.air import FailureConfig
from Teacher import get_args
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms.registry import POLICIES
from utils import round_to_multiple
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from ray.tune.utils import wait_for_gpu
import ray

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

class Trainable(tune.Trainable):
    def setup(self, config,teacher_args=None):


        self.stat_manager = statManager.remote((args.text_input_length,))


        """ fcnet_activation = 0.1823053454576449
        fcnet_hiddens_layer_count= 2.0
        layer_width= 738.0168809379319 """

        lstm_state_size = 256
        wait_for_gpu(target_util=0.75)
    
        lr = float(config["lr"])
        lambda_ = float(config["lambda_"])
        gamma = float(config["gamma"])
        grad_clip = float(config["grad_clip"])
        num_sgd_iter = float(config["num_sgd_iter"])
        sgd_minibatch_size = float(config["sgd_minibatch_size"])
        clip_param = float(config["clip_param"])
        vf_loss_coeff = float(config["vf_loss_coeff"])
        entropy_coeff = float(config["entropy_coeff"])
        fcnet_hiddens_layer_count = float(config["fcnet_hiddens_layer_count"])
        layer_width = float(config["layer_width"])
        fcnet_activation = float(config["fcnet_activation"])
        train_batch_size = float(config["train_batch_size"])
        #max_seq_len = float(config["max_seq_len"])
        #lstm_state_size = float(config["lstm_state_size"])

        num_sgd_iter = int(num_sgd_iter)
        #max_seq_len = round_to_multiple(max_seq_len,1,"up")
        sgd_minibatch_size = round_to_multiple(sgd_minibatch_size,128,"up")
        clip_param = round_to_multiple(clip_param, 0.1,"up")
        fcnet_hiddens_layer_count = round_to_multiple(fcnet_hiddens_layer_count,1,"up")
        layer_width = round_to_multiple(layer_width,128,"up")
        #lstm_state_size = round_to_multiple(layer_width,128,"up")
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
                "teacher_args":teacher_args,
                "stat_manager":self.stat_manager}

        config = PPOConfig(algo_class=MyPPO)


        config.train_batch_size = train_batch_size
        config.lr = lr
        config.grad_clip = grad_clip
        config.clip_param = clip_param
        config.sgd_minibatch_size = sgd_minibatch_size
        config.num_sgd_iter = num_sgd_iter
        config.model["fcnet_hiddens"] = hidden_layers
        config.model["fcnet_activation"] = fcnet_activation
        config.vf_loss_coeff = vf_loss_coeff
        config.entropy_coeff = entropy_coeff
        config.lambda_ = lambda_
        config.gamma = gamma
        #config.model["max_seq_len"] = max_seq_len

        config.model["custom_model"] = rlib_model
        config.model["custom_model_config"] = {"obs":args.text_input_length,
                                                "fc_size":layer_width,
                                                "lstm_state_size":lstm_state_size,
                                                "fc_layers_count":fcnet_hiddens_layer_count,
                                                }
        config.model["vf_share_layers"] = False

        config.framework(framework="torch")
        config.env = "UrsinaGym"
        config.disable_env_checking = False
        config.recreate_failed_workers= True
        config.restart_failed_sub_environments = True
        config.env_config.update(env_config)
        config.num_envs_per_worker = 7
        config.num_workers = 0
        config.remote_worker_envs = True  
        config.num_gpus = 0.5         
        config.num_gpus_per_worker = 0 
        config.num_cpus_per_worker = config.num_envs_per_worker
        config.remote_env_batch_wait_ms = 4
        config.vf_clip_param = 10

        config.batch_mode = "complete_episodes"
        config.horizon = 100
        config.log_level = "WARN"#"INFO"
        config.create_env_on_local_worker = True
        if config.create_env_on_local_worker:
            config.num_cpus_for_local_worker = config.num_envs_per_worker
        config.evaluation_interval = None

        self.algo = config.build()

    def step(self):
        return self.algo.train()

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        return self.algo.save_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        return self.algo.load_checkpoint(checkpoint)

    def reset_config(self, new_config: Dict):
        train_batch_size = float(new_config["train_batch_size"])
        sgd_minibatch_size = float(new_config["sgd_minibatch_size"])
        num_sgd_iter = float(new_config["num_sgd_iter"])


        train_batch_size = round_to_multiple(train_batch_size,1000,"up")
        sgd_minibatch_size = round_to_multiple(sgd_minibatch_size,128,"up")
        num_sgd_iter = int(num_sgd_iter)




        self.algo.config["train_batch_size"] = train_batch_size
        self.algo.config["sgd_minibatch_size"] = sgd_minibatch_size
        self.algo.config["num_sgd_iter"] = num_sgd_iter

        update_params = lambda policy,policyId: policy.update_params(new_config)

        self.algo.workers.foreach_policy_to_train(update_params)

        return True

    def cleanup(self):
        return
        self.algo.cleanup()
        ray.kill(self.stat_manager)

if __name__ == '__main__':
    args = parser.parse_args()

    register_env("UrsinaGym", lambda config: UrsinaGym(config))
    ModelCatalog.register_custom_model("rlib_model", rlib_model)
    #ModelCatalog.register_custom_model("rlib_model_lstm", rlib_model_lstm)
    tune.register_trainable("MyTrainable",Trainable)


    teacher_args = get_args()

    #teacher = Teacher.options(name="teacher").remote(teacher_args)


    config = {"lr":tune.uniform(1e-6,2e-4),
            "num_sgd_iter":tune.uniform(5,30),
            "sgd_minibatch_size":tune.uniform(65,20000),
            "clip_param":tune.uniform(0.01,0.3),
            "entropy_coeff":tune.uniform(0.0001,0.1),
            "layer_width":tune.uniform(32,2000),
            "vf_loss_coeff":tune.uniform(0.001,1),
            "fcnet_hiddens_layer_count":tune.uniform(0.5,4),
            "fcnet_activation":tune.uniform(0,1),
            "train_batch_size":tune.uniform(1025,20000),
            "grad_clip":tune.uniform(0.001,1),
            "lambda_":tune.uniform(0.95,1),
            "gamma":tune.uniform(0.9,0.99),

            } 
    #"max_seq_len":tune.uniform(0.5,20),
    #"lstm_state_size":tune.uniform(32,1000)
    """ config = {
            "clip_param": tune.choice([0.001, 0.01, 0.05]),
            "entropy_coeff": tune.choice([0.0001, 0.001, 0.005]),
            "lr": tune.choice([9e-5, 0.00015586661343287977, 4e-4]),
            "num_sgd_iter": tune.choice([12, 14, 16]),
            "sgd_minibatch_size": tune.choice([5000, 6000, 7000]),
            "train_batch_size": tune.choice([7000, 8000, 9000]),
            "vf_loss_coeff": tune.choice([0.001, 0.1, 0.2])
            } """

    """ trainable_with_resources  = tune.with_resources(
        tune.with_parameters(Trainable,teacher_args=teacher_args),tune.PlacementGroupFactory([
            {"CPU": 3, "GPU": 0.25},
            {"CPU": 3, "GPU": 0.25},


        ]))  """
    trainable_with_resources  = tune.with_resources(
        Trainable,tune.PlacementGroupFactory([
            {"CPU": 7, "GPU": 0.5},
            {"CPU": 1},
        ])) 

    stopper = CustomStopper()



    """ pbt = PB2(
        time_attr="timesteps_total",
        perturbation_interval=50000,
        synch=True,
        hyperparam_bounds={
            "num_sgd_iter": [5,30],
            "sgd_minibatch_size": [65,10000],
            "train_batch_size": [1025,10000],
            "clip_param":[0.001,0.3],
            "lr":[1e-6,1e-4],
            "entropy_coeff":[0.0001,0.1],
            "vf_loss_coeff":[0.001,0.9]
        }
    ) """
    

    """ tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            mode="max",
            metric="episode_reward_mean",
            scheduler=pbt,
            num_samples=4,
            reuse_actors=True
        ),
        run_config=RunConfig(
            verbose=3,
            stop=stopper,
            failure_config=FailureConfig(
                fail_fast=True),

        ),

    ) """


    search_alg=BayesOptSearch(
            random_search_steps=15
            )

    scheduler=AsyncHyperBandScheduler(
            time_attr="timesteps_total",
            grace_period=200000,
            max_t=700000
            )
    tuner = tune.Tuner(
            trainable_with_resources,
            param_space=config,
            tune_config=tune.TuneConfig(
                mode="max",
                metric="episode_reward_mean",
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=35,
                max_concurrent_trials=2
            ),
            run_config=RunConfig(
                verbose=3,
                stop=stopper,
                log_to_file=True,
                failure_config=FailureConfig(
                    fail_fast=True
                    ),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                    checkpoint_frequency=1,
                    checkpoint_score_attribute="episode_reward_mean"
                    )

            ),

        )
    results = tuner.fit()  

    """ tune.run(resume="LOCAL",
    fail_fast=True,
    checkpoint_at_end=True,
    local_dir=r"C:\Users\sohai\ray_results",
    name="Trainable_2023-01-14_16-51-39",
    run_or_experiment=trainable_with_resources,
    config=config,
    checkpoint_freq=1,
    keep_checkpoints_num=1,
    checkpoint_score_attr="episode_reward_mean",
    verbose=3,
    stop=stopper,
    log_to_file=True,
    mode="max",
    metric="episode_reward_mean",
    num_samples=30,
    max_concurrent_trials=2,
    scheduler=scheduler,
    search_alg=search_alg)  """

    #tuner = tune.Tuner.restore(r"C:\Users\sohai\ray_results\Trainable_2023-01-14_16-51-39")
    #results = tuner.fit()

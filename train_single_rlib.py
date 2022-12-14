from ray.tune.registry import register_env
from gym_ursina import make_env
import argparse
from ray.rllib.models import ModelCatalog
from models import rlib_model,MyPPO,CustomStopper,Teacher,rlib_model_lstm,statManager
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

    
        lr = float(config["lr"])
        num_sgd_iter = float(config["num_sgd_iter"])
        sgd_minibatch_size = float(config["sgd_minibatch_size"])
        clip_param = float(config["clip_param"])
        vf_loss_coeff = float(config["vf_loss_coeff"])
        entropy_coeff = float(config["entropy_coeff"])
        fcnet_hiddens_layer_count = float(config["fcnet_hiddens_layer_count"])
        layer_width = float(config["layer_width"])
        fcnet_activation = float(config["fcnet_activation"])
        train_batch_size = float(config["train_batch_size"])
        

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
                "obs_size":args.text_input_length,
                "mask_size":29,
                "min_size":75,
                "max_size":250,
                "is_teacher":True,
                "teacher_args":teacher_args}

        config = PPOConfig(algo_class=MyPPO)

        """ curiosty_model_config = config.model.copy()
        curiosty_model_config["fcnet_hiddens"] = [512,512] """

        config.framework(framework="torch")
        config.env = "UrsinaGym"
        config.disable_env_checking = False
        config.recreate_failed_workers= True
        config.restart_failed_sub_environments = True
        config.env_config.update(env_config)
        config.num_envs_per_worker = 4
        config.num_workers = 3
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
        config.model["vf_share_layers"] = True
        config.vf_loss_coeff = vf_loss_coeff
        config.entropy_coeff = entropy_coeff
        config.model["custom_model"] = rlib_model_lstm
        config.model["custom_model_config"] = {"obs":args.text_input_length,
                                                "fc_size":layer_width,
                                                "lstm_state_size":layer_width}
        config.model["max_seq_len"] = 10
        config.horizon = 100
        config.log_level = "WARN"#"INFO"
        config.create_env_on_local_worker = False
        if config.create_env_on_local_worker:
            config.num_cpus_for_local_worker = config.num_envs_per_worker

        """ config.exploration(explore=True,
                            exploration_config={
                                "type":"Curiosity",
                                "feature_net_config":curiosty_model_config,
                                "sub_exploration":{"type":"StochasticSampling"}
                            }) """
                            
        self.algo = config.build()

    def step(self):
        return self.algo.train()

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        statManager = ray.get_actor("statManager")
        statManager.save_stat.remote()
        return self.algo.save_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        return self.algo.load_checkpoint(checkpoint)



if __name__ == '__main__':
    args = parser.parse_args()

    register_env("UrsinaGym", lambda config: UrsinaGym(config))
    ModelCatalog.register_custom_model("rlib_model", rlib_model_lstm)
    tune.register_trainable("MyTrainable",Trainable)


    teacher_args = get_args()

    config = {
        "clip_param": 0.01,
        "entropy_coeff": 0.001,
        "fcnet_activation": 0.1823053454576449,
        "fcnet_hiddens_layer_count": 2.0,
        "layer_width": 738.0168809379319,
        "lr": 0.00015586661343287977,
        "num_sgd_iter": 13.394627394465651,
        "sgd_minibatch_size": 5398.568478435033,
        "train_batch_size": 7583.765925047164,
        "vf_loss_coeff": 0.1
            }
        #"train_batch_size": 7583.765925047164,

    """ trainable_with_resources  = tune.with_resources(
        tune.with_parameters(Trainable,teacher_args=teacher_args),tune.PlacementGroupFactory([
            {"CPU": 1, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25}
        ]))   """
    """ trainable_with_resources  = tune.with_resources(
        tune.with_parameters(Trainable,teacher_args=teacher_args),tune.PlacementGroupFactory([
            {"CPU": 13, "GPU": 1}
        ]))  """

    trainable_with_resources  = tune.with_resources(
        tune.with_parameters(Trainable,teacher_args=teacher_args),tune.PlacementGroupFactory([
            {"CPU": 1, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25},
            {"CPU":4, "GPU": 0.25},

        ]))  
    #stopper = CustomStopper()

    teacher = Teacher.options(name="teacher").remote(teacher_args)
    stat_manager = statManager.options(name="statManager").remote((args.text_input_length,))


    result = tune.run(
        run_or_experiment=trainable_with_resources,
        config=config,
        local_dir=r"C:\Users\sohai\ray_results\tensorboard",
        checkpoint_at_end=True,
        log_to_file=True,
        checkpoint_freq=4,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean"
        )  
 

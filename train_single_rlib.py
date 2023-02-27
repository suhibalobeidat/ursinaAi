from ray.tune.registry import register_env
from gym_ursina import make_env
import argparse
from ray.rllib.models import ModelCatalog
from models import rlib_model,CustomStopper,Teacher,rlib_model_lstm,statManager
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

        self.stat_manager = statManager.options(name="statManager").remote((args.text_input_length,))

        dir = r"C:\Users\sohai\Desktop\data_stat\Trainable_bde87636_5_clip_param=0.2825,entropy_coeff=0.0895,fcnet_activation=0.5979,fcnet_hiddens_layer_count=4.6484,gamma=0.8168,3"
        self.stat_manager.save_stat.remote(dir,"data_stat.h5")

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
                "teacher_args":teacher_args,
                "stat_manager":None}

        config = PPOConfig(algo_class=PPO)

        """ curiosty_model_config = config.model.copy()
        curiosty_model_config["fcnet_hiddens"] = [512,512] """

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
        #config.model["max_seq_len"] = 20

        config.model["custom_model"] = rlib_model
        config.model["custom_model_config"] = {"obs":args.text_input_length,
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
        config.num_envs_per_worker = 7
        config.num_rollout_workers = 1
        config.remote_worker_envs = True  
        config.num_gpus = 0.5         
        config.num_gpus_per_worker = 0.5 
        config.num_cpus_per_worker = config.num_envs_per_worker
        config.remote_env_batch_wait_ms = 4
        config.vf_clip_param = 10

        config.kl_coeff = 0
        config.batch_mode = "complete_episodes"
        config.horizon = 100
        config.log_level = "WARN"#"INFO"
        config.create_env_on_local_worker = True
        if config.create_env_on_local_worker:
            config.num_cpus_for_local_worker = config.num_envs_per_worker
        config.evaluation_interval = None

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
        dir = r"C:\Users\sohai\Desktop\data_stat\Trainable_bde87636_5_clip_param=0.2825,entropy_coeff=0.0895,fcnet_activation=0.5979,fcnet_hiddens_layer_count=4.6484,gamma=0.8168,3"
        self.stat_manager.save_stat.remote(dir,"data_stat.h5")
        return self.algo.save_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        return self.algo.load_checkpoint(checkpoint)



if __name__ == '__main__':
    args = parser.parse_args()

    register_env("UrsinaGym", lambda config: UrsinaGym(config))
    ModelCatalog.register_custom_model("rlib_model", rlib_model)
    tune.register_trainable("MyTrainable",Trainable)


    teacher_args = get_args()

    config = {
        "clip_param": 0.2824546930536148,
        "entropy_coeff": 0.08949325230772212,
        "fcnet_activation": 0.5978999788110851,
        "fcnet_hiddens_layer_count": 4.6484340576040255,
        "gamma": 0.8168135753898648,
        "grad_clip": 0.19678687955672605,
        "lambda_": 0.9522613644455269,
        "layer_width": 672.2500909421042,
        "lr": 7.834678064820692e-05,
        "num_sgd_iter": 10.426980635477918,
        "sgd_minibatch_size": 16585.88224494371,
        "train_batch_size": 7794.394374010856,
        "vf_loss_coeff": 0.2816535751776934
    } 

    """ config = {
    "clip_param": 0.176192007866709,
    "entropy_coeff": 0.006033957647477133,
    "fcnet_activation": 0.595773002166077,
    "fcnet_hiddens_layer_count": 1.7777498316417781,
    "gamma": 0.9541755115840513,
    "grad_clip": 0.2512786625817104,
    "lambda_": 0.9531417614596301,
    "layer_width": 1041.6393734105818,
    "lr": 7.201914539290346e-05,
    "num_sgd_iter": 26.105040824428173,
    "sgd_minibatch_size": 18023.642141369033,
    "train_batch_size": 10096.024045377108,
    "vf_loss_coeff": 0.5135697404733504
    } """

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
            {"CPU": 7, "GPU": 0.5},
            {"CPU":7, "GPU": 0.5},
            {"CPU":1}
        ]))  
    #stopper = CustomStopper()

    #teacher = Teacher.options(name="teacher").remote(teacher_args)
    #stat_manager = statManager.options(name="statManager").remote((args.text_input_length,))

    #stat_manager.load_stat.remote()


    result = tune.run(
        resume=False,
        run_or_experiment=trainable_with_resources,
        config=config,
        checkpoint_at_end=True,
        log_to_file=True,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
        num_samples=1
        )   
 

import random

from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.air.config import RunConfig,CheckpointConfig
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms.ppo.ppo import PPO,PPOConfig


class MyPPO2(PPO):
    def reset_config(self, new_config):
        return True


if __name__ == "__main__":


    tune.register_trainable("MyPPO2",MyPPO2)

    pbt = PB2(
        time_attr="training_iteration",
        perturbation_interval=5,

        hyperparam_bounds={
            "lambda": [0.9, 1.0],
            "clip_param": [0.01, 0.5],
            "lr": [1e-3,1e-5],
            "num_sgd_iter": [1, 30],
            "sgd_minibatch_size": [128, 2000],
            "train_batch_size": [2000, 6000],
        },
    ) 
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    """ pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 2000),
            "train_batch_size": lambda: random.randint(2000, 4000),
        },
        custom_explore_fn=explore    ) """
    """ tuner = tune.Tuner(
        "MyPPO2",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=4,
            reuse_actors=True
        ),
        run_config=RunConfig(verbose=3,
        checkpoint_config=CheckpointConfig(num_to_keep=4,
        checkpoint_score_attribute = "episode_reward_mean",
        checkpoint_frequency=5)), 
        param_space={
            "env": "CartPole-v1",
            "kl_coeff": 1.0,
            "num_workers": 2,
            "num_envs_per_worker":4,
            "num_gpus": 0.25, # number of GPUs to use
            "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-4,
            "grad_clip":0.5,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([128, 512, 2000]),
            "train_batch_size": tune.choice([2000, 3000, 4000]),
            "framework":"torch",
            "evaluation_interval":None  
        },
    )  """

    tuner = tune.Tuner.restore(r"C:\Users\sohai\ray_results\MyPPO2",restart_errored=True)
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
import gym
from gym import spaces
import numpy as np
from env_ursina import env_interface
import random
from Teacher import get_teacher
from gym.wrappers.normalize import NormalizeReward, NormalizeObservation
import ray
class InnerEnv(gym.Env):
    def __init__(self,env_config):

        mask_size = env_config["mask_size"]
        obs_size = env_config["obs_size"]
        min_size = env_config["min_size"]
        max_size = env_config["max_size"]
        is_teacher = env_config["is_teacher"]

        self.mask_size = mask_size

        self.observation_space = spaces.Box(-100., 100., shape=(obs_size,))
        self.action_space = spaces.Discrete(self.mask_size)
        self.is_teacher = is_teacher

        if self.is_teacher:
            self.teacher = ray.get_actor("teacher")#get_teacher(env_config["teacher_args"])

        self.env = env_interface(min_size,max_size,same_process=True)
        self.env.init()


    def reset(self,**kwargs):

        if self.is_teacher:
            self.params = ray.get(self.teacher.get_env_params.remote())
            self.teacher_ret = 0
            self.steps = 0
        else:
            self.params = np.array([random.random() for i in range(3)])

        receivedData = self.env.reset(self.params)
        observation = np.array(receivedData[:len(receivedData)-self.mask_size],dtype=np.float32)
        action_mask = np.array(receivedData[len(receivedData)-self.mask_size:],dtype=np.float32)

        info = {"action_mask":action_mask}

        return observation,info

    def step(self, action):

        receivedData = self.env.step(action)
        observation = np.array(receivedData[:len(receivedData)-(self.mask_size+2)],dtype=np.float32)
        action_mask = np.array(receivedData[len(receivedData)-(self.mask_size+2):len(receivedData)-2],dtype=np.float32)
        done = receivedData[len(receivedData)-2]
        reward = receivedData[len(receivedData)-1]

        if done == 0:
            terminated = False
            truncated = True
        elif done == 1:#collide
            terminated = True
            truncated = False
        elif done == 2:#max iteration exceeded
            terminated = True 
            truncated = True
        elif done == 3:#successful
            terminated = True
            truncated = False



        if self.is_teacher:
            gamma = pow(0.99,self.steps)
            self.teacher_ret += gamma * reward
            self.steps +=1
            if terminated or truncated:
                self.teacher.record_train_episode.remote(self.teacher_ret, 0,self.params)

        info = {"truncated":truncated,"action_mask":action_mask}

        return observation, reward, terminated, info

    def render(self):
        return None

    def close(self):
        self.env.close()
class UrsinaGym(gym.Env):
    def __init__(self,env_config):

        mask_size = env_config["mask_size"]
        obs_size = env_config["obs_size"]
        
        self.observation_space = self.observation_space = spaces.Dict({
            "obs": spaces.Box(-10., 10., shape=(obs_size,)),
            "action_mask": spaces.Box(0, 1, shape=(mask_size, ))
        })
        self.action_space = spaces.Discrete(mask_size)

        """ self.statManager = None
        if env_config["stat_manager"]:
            self.statManager = env_config["stat_manager"] """
        
        self.statManager = ray.get_actor("statManager")

        self.inner_env = InnerEnv(env_config)

        self.obs = []
        self.rews = []

        self.obs_mean = np.zeros((obs_size,), "float64")
        self.obs_var =  np.ones((obs_size,), "float64")
        self.ret_var = np.ones((), "float64")

        self.epsilon=1e-4
        self.clipob = 10
        self.cliprew = 10

    def reset(self):      
        self.obs = []
        self.rews = []
        obs,info = self.inner_env.reset(return_info=True)

        self.obs.append(obs)

        obs = np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon), -self.clipob, self.clipob)

        observation = {"obs":obs,"action_mask":info["action_mask"]}


        return observation

    def step(self, action):
        obs, reward, terminated, info =  self.inner_env.step(action)
        

        self.obs.append(obs)
        self.rews.append(reward)
        
        if self.statManager:
            if terminated:
                self.obs_mean,self.obs_var,_,self.ret_var = ray.get(self.statManager.update_mean_var.remote(np.stack(self.obs),self.rews))

        obs = np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon), -self.clipob, self.clipob)

        reward = np.clip(reward / np.sqrt(self.ret_var + self.epsilon), -self.cliprew, self.cliprew)
        
        observation = {"obs":obs,"action_mask":info["action_mask"]}

        return observation, reward, terminated, info


    def render(self):
        return None

    def close(self):
        self.inner_env.close()


def make_env(env_config = None)-> gym.Env:
    env = UrsinaGym(env_config)
    return env
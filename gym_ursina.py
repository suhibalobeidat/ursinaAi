import gym
from gym import spaces
import numpy as np
from env_ursina import env_interface
import random
from Teacher import get_teacher
from gym.wrappers.normalize import NormalizeReward, NormalizeObservation

class UrsinaGym(gym.Env):
    def __init__(self,env_config):

        

        mask_size = env_config["mask_size"]
        obs_size = env_config["obs_size"]
        min_size = env_config["min_size"]
        max_size = env_config["max_size"]
        is_teacher = env_config["is_teacher"]

        self.mask_size = mask_size

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-100., 100., shape=(obs_size,)),
            "action_mask": spaces.Box(0, 1, shape=(mask_size, ))
        })

        #self.observation_space = spaces.Box(-100., 100., shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.mask_size)
        self.is_teacher = is_teacher

        if self.is_teacher:
            self.teacher = get_teacher()

        self.env = env_interface(min_size,max_size,same_process=True)
        self.env.init()

        #print("INIT IS DONE")


    def reset(self):
        # We need the following line to seed self.np_random

        if self.is_teacher:
            self.params = self.teacher.get_env_params()
            self.teacher_ret = 0
        else:
            self.params = np.array([random.random() for i in range(2)])



        receivedData = self.env.reset(self.params)

        obs = np.array(receivedData[:len(receivedData)-self.mask_size],dtype=np.float32)
        action_mask = np.array(receivedData[len(receivedData)-self.mask_size:],dtype=np.float32)

        observation = {"obs":obs,"action_mask":action_mask}

        #print("RESET IS DONE")

        return observation

    def step(self, action):

        receivedData = self.env.step(action)

        obs = np.array(receivedData[:len(receivedData)-(self.mask_size+2)],dtype=np.float32)
        action_mask = np.array(receivedData[len(receivedData)-(self.mask_size+2):len(receivedData)-2],dtype=np.float32)
        done = receivedData[len(receivedData)-2]
        reward = receivedData[len(receivedData)-1]

        if done == 2:
            terminated = True 
            truncated = True
        elif done == 3:
            terminated = True
            truncated = False
        else:
            terminated = bool(done)
            truncated = False

        observation = {"obs":obs,"action_mask":action_mask}


        if self.is_teacher:
            self.teacher_ret += 0.99 * reward
            if terminated or truncated:
                self.teacher.record_train_episode(self.teacher_ret, 0,self.params)

        info = {"truncated":truncated}
        #print("STEP IS DONE")

        return observation, reward, terminated, info

    def render(self):
        return None


def make_env(env_config = None)-> gym.Env:
    env = UrsinaGym(env_config)
    #env = NormalizeReward(env)
    #env = NormalizeObservation(env)

    return env
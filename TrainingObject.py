
import numpy as np

class TrainingObject_parallel():
    def __init__(self, parallel_workers, batch_first = True):
        self.parallel_workers = parallel_workers
        self.batch_first = batch_first
        self.workers  = [[] for i in range(parallel_workers)]
  

    def append(self,object, add):
        for i in range(self.parallel_workers):
            if add[i] != -1:
                if self.batch_first:
                    self.workers[i].append(object[i])
                else:
                    self.workers[i].append(object[:,i,:])

    def get_next_value(self,next_value,done):
        next_values = []
        for i in range(self.parallel_workers):
            if done[i] == 1:
                next_values.append(next_value[i])
        return next_values

    
    def get_completed(self,done):
        completed_episodes = []
        for i in range(self.parallel_workers):
            if done[i] == 1:
                if len(self.workers[i]) != 0:
                    completed_episodes.append(self.workers[i].copy())
                    self.workers[i].clear()
        return completed_episodes

    def compute_gae(self,next_value,reward, terminals, value, gamma=0.99, tau=0.95):
        completed_episodes_returns = []
        completed_episodes_advantages = []

        for i in range(len(value)):
            values = value[i].copy()
            values.append(next_value[i])
            masks = terminals[i]
            rewards = reward[i]
            gae = 0
            returns = []
            advantages = []
            for step in reversed(range(len(rewards))):
                retrn = rewards[step] + gamma * values[step + 1] * masks[step]
                delta = retrn - values[step]
                gae = delta + gamma * tau * masks[step] * gae
                returns.insert(0,gae+values[step])
                advantages.insert(0,gae)
            completed_episodes_returns.append(returns)
            completed_episodes_advantages.append(advantages)
            
        return completed_episodes_returns,completed_episodes_advantages
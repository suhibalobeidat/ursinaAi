import numpy as np


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = np.full((), epsilon, 'float64')

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_Images(self,images):
        batch_mean = np.mean(images,axis=(0, 1, 2))  # Take the mean over the batch,H,W axes
        batch_var = np.var(images,axis=(0, 1, 2))  # Take the variance over the batch,H,W axes
        batch_count = images.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv:
    def __init__(self, obs_shape,parallel_workers,depth_map_length, ob=True, ret=True,im=True, clipob=10.0, cliprew=10.0, gamma=0.99, epsilon=1e-8):
        self.ob_rms = RunningMeanStd(shape=obs_shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.im_rms = RunningMeanStd(shape=(3,)) if im else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(parallel_workers)
        self.gamma = gamma
        self.epsilon = epsilon
        self.depth_map_length = depth_map_length

    def step(self, obs, rewards,done, Test = False):

        if self.depth_map_length != 0:
            depth_map = obs[:][0:self.depth_map_length]

        
        new_obs = []
        indexs = []
        new_dones = []

        for i in range(done.shape[0]):
            if done[i] != -1:
                new_obs.append(obs[i])
                indexs.append(i)
                new_dones.append(1)
            else:
                new_dones.append(0)

        new_obs = np.stack(new_obs,axis=0)
        
        new_obs = self._obfilt(new_obs)

        for i in range(len(indexs)):
            obs[indexs[i]] = new_obs[i]

        if self.depth_map_length != 0:
            obs[:][0:self.depth_map_length] = depth_map

        rews = rewards.flatten()

        if self.ret_rms:
            if not Test:
                self.ret = (self.ret * self.gamma + rews) * np.array(new_dones)
                new_ret = np.delete(self.ret, np.where(self.ret == 0))

                self.ret_rms.update(new_ret)
                rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews.reshape(-1,1)

    def _obfilt(self, obs, Test = False):
        if self.ob_rms:
            if not Test:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def normalizeImage(self,images,Test=False):
        if self.im_rms:
            if not Test:
                self.im_rms.update(images)
            images = np.clip((images - self.im_rms.mean) / np.sqrt(self.im_rms.var + self.epsilon), -self.clipob, self.clipob)
            return images
        else:
            return images


    def normalize_image(self,images,dones,Test = False):      
        new_images = []
        indexs = []

        for i in range(dones.shape[0]):
            if dones[i]:
                new_images.append(images[i])
                indexs.append(i)

        new_images = np.stack(new_images,axis=0)
        new_images = self.normalizeImage(new_images)

        for i in range(len(indexs)):
            images[indexs[i]] = new_images[i]

        return images


    def reset(self, obs,mask,dones, Test = False):
        
        if not Test:
            self.ret = (1-dones).flatten() * self.ret

        new_obs = []
        new_mask = []
        indexs = []

        if self.depth_map_length != 0:
            depth_map = obs[:][0:self.depth_map_length]

        for i in range(dones.shape[0]):
            if dones[i]:
                new_obs.append(obs[i])
                new_mask.append(mask[i])
                indexs.append(i)

        new_obs = np.stack(new_obs,axis=0)
        
        new_obs = self._obfilt(new_obs)

        for i in range(len(indexs)):
            obs[indexs[i]] = new_obs[i]
            mask[indexs[i]] = new_mask[i]
            
        if self.depth_map_length != 0:
            obs[:][0:self.depth_map_length] = depth_map


        return obs,mask
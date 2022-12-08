import numpy as np
import h5py
import torch
import os.path
import gym
from math import ceil, floor

def get_data_statistics(file_name):
    file = h5py.File(file_name, "r+")


    texts_mean = np.array(file["/texts_mean"]).astype("float32")
    texts_std = np.array(file["/texts_std"]).astype("float32")


    print(f"texts_mean {texts_mean.shape}")
    print(f"texts_std {texts_std.shape}")


    return 127.5,127.5,texts_mean,texts_std

def check_done(done):
    done_bool = np.empty((done.shape[0],1))
    successful_ep = np.full((done.shape[0], 1), 0)
    for i in range(len(done)):
        if done[i] == 2:
            done_bool[i] = 0 
            done[i] = 1
        elif done[i] == 3:
            done_bool[i] = 1
            done[i] = 1
            successful_ep[i] = 1
        else:
            done_bool[i] = float(done[i])
    return done_bool,successful_ep


def images_args():
    import argparse
    parser = argparse.ArgumentParser(description='state images')
    parser.add_argument('--planimage', default='C:/Users/sohai/Desktop/GAIL/planimages/*.png', type=str, help='Input filename or folder.')
    parser.add_argument('--initplanimage', default='C:/Users/sohai/Desktop/GAIL/initplanimages/*.png', type=str, help='Input filename or folder.')
    parser.add_argument('--testImages', default='C:/Users/sohai/Desktop/GAIL/img/*.png', type=str, help='Input filename or folder.')

    args = parser.parse_args()
    return args


def save_model(model ,filename, directory):
    torch.save(model, '%s/%s.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
def load_model(_model,filename, directory):
    _model.load_state_dict(torch.load('%s/%s.pth' % (directory, filename)))


def create_data_stat(dir,texts_mean,texts_std):
    file = h5py.File(dir + "/data_stat.h5", "w")

    file.create_dataset(
            "texts_mean", np.shape(texts_mean), data=texts_mean
        )
    file.create_dataset(
            "texts_std", np.shape(texts_std), data=texts_std
        )
    file.close()


def round_to_multiple(number, multiple, direction='nearest'):
    if direction == 'nearest':
        return multiple * round(number / multiple)
    elif direction == 'up':
        return multiple * ceil(number / multiple)
    elif direction == 'down':
        return multiple * floor(number / multiple)
    else:
        return multiple * round(number / multiple)
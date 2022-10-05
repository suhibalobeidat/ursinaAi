import torch
import os
import argparse
from pathlib import Path
import numpy as np
import random
from vec_normalize_parallel import NormalizedEnv
from train_utils import test_ursina
from utils import images_args,load_model,get_data_statistics
from models import ActorCritic
import torch.backends.cudnn
from counter import Counter
from env_ursina import parallel_envs
import warnings

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--text_input_length', type=int, default=413, help='406 Number of features in text input')
parser.add_argument('--depth_map_length', type=int, default=361, help='361 Number of features in text input')
parser.add_argument('--action_direction_length', type=int, default=29, help='possible actions')
parser.add_argument('--max_action_length', type=int, default=10, help='the max action length')
parser.add_argument('--seed', type=int, default=7, help='seed to initialize libraries')
parser.add_argument('--load_model', type=bool, default=True, help='load a pretrained model')
parser.add_argument('--parallel_workers_test', type=int, default=1, help='number of parallel agents')




if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    warnings.filterwarnings('ignore')

    args = parser.parse_args()
    image_args = images_args()

    torch.random.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_trained_file_name = "%s_%s_%s" % ("model", "pipeit_navigation", str(args.seed))

    model = ActorCritic(args.text_input_length,args.depth_map_length,args.action_direction_length).to(device)
    
    normalizedEnv = NormalizedEnv(np.zeros(shape=(1,args.text_input_length)).shape,args.parallel_workers_test,args.depth_map_length)

    if args.load_model:
        load_model(_model = model,filename = "%s" % (ppo_trained_file_name), directory="./pytorch_models")
        images_mean,images_std,texts_mean,texts_std = get_data_statistics(Path("data_stat.h5"))
        data_stat ={'images_mean':127.5,'images_std':127.5,'texts_mean':texts_mean,'texts_std':texts_std}
        normalizedEnv.ob_rms.mean = texts_mean
        normalizedEnv.ob_rms.var = np.power(texts_std,2)
        normalizedEnv.ob_rms.count = 20000 

    env = parallel_envs(args.parallel_workers_test,args.text_input_length,args.action_direction_length)

    counter = Counter()

    test_ursina(model,env,None,normalizedEnv,counter,args,image_args,device)

    print("AVERAGE REWARD", counter.test_average_reward)
    print("AVERAGE STEPS", counter.test_average_steps)
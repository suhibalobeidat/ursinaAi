import time
import torch
import torch.optim as optim
import os
import argparse
from pathlib import Path
import numpy as np
import random
from Teacher import get_teacher
from vec_normalize_parallel import NormalizedEnv
from train_utils import ppo_update,get_epoch_trajectories_ursina,test_ursina,to_torch_tensor,collector
from utils import images_args,load_model,get_data_statistics,save_model,create_data_stat
from torch.utils.tensorboard import SummaryWriter
from models import ActorCritic
import torch.backends.cudnn
from counter import Counter
from env_ursina import parallel_envs,mul_parallel_envs
import warnings
from gpu_usage import print_gpu_memory_every_5secs
from modified_thread import thread_with_exception
import queue

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Learning rate for discriminator')
parser.add_argument('--critic_loss_coeff', type=float, default=0.5, help='Learning rate for discriminator')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Learning rate for discriminator')
parser.add_argument('--bs', type=int, default=200, help='Batch size')
parser.add_argument('--ppo_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--text_input_length', type=int, default=413, help='406 Number of features in text input')
parser.add_argument('--depth_map_length', type=int, default=361, help='361 Number of features in text input')
parser.add_argument('--action_direction_length', type=int, default=29, help='possible actions')
parser.add_argument('--max_action_length', type=int, default=10, help='the max action length')
parser.add_argument('--num_steps', type=int, default=200, help='number of steps per epoch')
parser.add_argument('--test_steps', type=int, default=200, help='number of steps per epoch')
parser.add_argument('--seed', type=int, default=7, help='seed to initialize libraries')
parser.add_argument('--max_iter', type=int, default=3000000, help='max number of steps')
parser.add_argument('--load_model', type=bool, default=False, help='load a pretrained model')
parser.add_argument('--compute_dynamic_stat', type=bool, default=True, help='collect the agents data in parallel')
parser.add_argument('--anneal_lr', type=bool, default= False, help='collect the agents data in parallel')
parser.add_argument('--parallel_workers_test', type=int, default=10, help='number of parallel agents')
parser.add_argument('--parallel_workers', type=int, default=4, help='number of parallel agents')
parser.add_argument('--envs_per_worker', type=int, default=3, help='number of parallel agents')
parser.add_argument('--sageMaker', type=bool, default=False, help='number of parallel agents')






def train_ppo(env,model,optimizer,normalizedEnv,writer,teacher,counter,image_args,args,device,ppo_trained_file_name,data_stat,model_dir):
   
    update_lr = lambda f: f * args.lr
    num_updates = 1000000 // args.num_steps
    update = 1

    gamma = 0.99
    decay = 0.5**(1./20)

    envs = []
    norm_envs = []
    for i in range(args.parallel_workers):
        env = parallel_envs(args.envs_per_worker,args.text_input_length,args.action_direction_length)
        envs.append(env)
        normalizedEnv = NormalizedEnv(np.zeros(shape=(1,args.text_input_length)).shape,args.envs_per_worker,args.depth_map_length)
        norm_envs.append(normalizedEnv)

    test_env = mul_parallel_envs(envs,args.parallel_workers_test,args.text_input_length,args.action_direction_length)
  
    worker_threads = []
    model_queues = []
    q_traj = queue.Queue(maxsize=args.parallel_workers)
    for i in range(args.parallel_workers):
        q_model = queue.Queue()
        q_model.put(model.state_dict())
        model_queues.append(q_model)
        thread = thread_with_exception(target=get_epoch_trajectories_ursina, args=(None,envs[i],teacher,norm_envs[i],counter,writer,device,args,q_model,q_traj))
        thread.start()
        worker_threads.append(thread)
        

    while True:
           
        counter.epochs_counter +=1

        counter.iter_successful_ep = 0
        counter.iter_epoch = 0


        epoch_start_time = time.time()

        traj_coll_start_time = time.time()

        #traj = get_epoch_trajectories_ursina(model,env,teacher,normalizedEnv,counter,writer,device,args,image_args)
        
        traj = collector(q_traj)

        iter_idx = counter.iter_index

        traj_coll_end_time = time.time()
        total_traj_coll_time = traj_coll_end_time - traj_coll_start_time
        writer.add_scalar('Trajectories collection time in s', total_traj_coll_time,iter_idx)
        print(f'time for trajectories collection is (ms):{(total_traj_coll_time)*1000}')
        print("[FINISH COLLECTING TRAJECTORIES]")  


        if args.anneal_lr:
                if update < num_updates:
                    frac = 1.0 - (update - 1.0) / num_updates
                    lrnow = update_lr(frac)
                    optimizer.param_groups[0]["lr"] = lrnow
                    update +=1
                    writer.add_scalar('learning rate',lrnow,iter_idx)



        if counter.iter_test >= args.test_steps:
            testing_start_time = time.time()

            counter.iter_test = 0
            counter.test_average_reward = 0
            counter.test_average_steps = 0
            test_ursina(model,test_env,teacher,norm_envs[0],counter,args,image_args,device)
            writer.add_scalar('Average task reward',(counter.test_average_reward),iter_idx)
            writer.add_scalar('Average number of steps',(counter.test_average_steps),iter_idx)

            testing_end_time = time.time()
            testing_total_time = testing_end_time - testing_start_time
            writer.add_scalar('testing time in s', testing_total_time,iter_idx)
            print(f'time for testing is (ms):{(testing_total_time)*1000}')


        print(f"ITERATION INDEX {counter.iter_index}")
        print("TOTAL COLLECTED EPISODES ",len(traj["text_inputs"]))
        writer.add_scalar('Collected episodes',len(traj["text_inputs"]),iter_idx)
        print("successful_ep", counter.iter_successful_ep)
        writer.add_scalar('successful_ep',counter.iter_successful_ep/(len(traj["text_inputs"])),iter_idx)
        
        if normalizedEnv.ob_rms:
            create_data_stat(model_dir,normalizedEnv.ob_rms.mean, np.sqrt(normalizedEnv.ob_rms.var))

        text_input = to_torch_tensor(traj["text_inputs"])
        actions_length = to_torch_tensor(traj["actions_length"])
        actions_directions = to_torch_tensor(traj["actions_directions"])
        actions_length_log_probs = to_torch_tensor(traj["actions_length_log_probs"])
        actions_directions_log_probs = to_torch_tensor(traj["actions_directions_log_probs"])
        actions_mask = to_torch_tensor(traj["actions_mask"])
        values = to_torch_tensor(traj["values"])
        returns = to_torch_tensor(traj["returns"])
        advantages = to_torch_tensor(traj["advantages"])
        advantages = (advantages-advantages.mean())/(advantages.std() + 1e-8)

        hidden_states = to_torch_tensor(traj["hidden_states"],1)
        cell_states = to_torch_tensor(traj["cell_states"],1)

        print(f"COLLECTED DATA {text_input.shape[0]}")


        print("[START TRAINING PPO]")
        ppo_training_start_time = time.time()

        writer.add_scalar("gamma",gamma,iter_idx)

  
        ppo_update(device,gamma,args.ppo_epochs, args.bs,text_input,actions_length,actions_directions,
            actions_length_log_probs,actions_directions_log_probs,returns,advantages,
            actions_mask,values,hidden_states, cell_states,counter,optimizer,model,writer,args)


        for q in model_queues:
            q.put(model.state_dict())

        ppo_training_end_time = time.time()
        total_ppo_training_time =  ppo_training_end_time - ppo_training_start_time  
        print(f'time for PPO training is (ms):{(total_ppo_training_time)*1000}')
        writer.add_scalar('PPO Training time in s',total_ppo_training_time,iter_idx)
        print("[finish TRAINING PPO]")


        gamma *= decay

        writer.add_histogram('advantages,',advantages,iter_idx)
        writer.add_histogram('returns,',returns,iter_idx)
        writer.add_histogram('actions_length,',actions_length,iter_idx)
        writer.add_histogram('actions_directions,',actions_directions,iter_idx)
        writer.add_histogram('values,',values,iter_idx)
        writer.add_histogram('actions_direction_log_probs,',actions_directions_log_probs,iter_idx)
        writer.add_histogram('actions_length_log_probs,',actions_length_log_probs,iter_idx)
  

        
        save_model(model = model.state_dict(),filename = "%s" % (ppo_trained_file_name), directory=model_dir)
        
        epoch_end_time = time.time()
        total_epoch_time = epoch_end_time - epoch_start_time
        print(f'time for epoch is (ms):{(total_epoch_time)*1000}')
        writer.add_scalar('Epoch time in s',total_epoch_time,iter_idx) 

        counter.total_time +=total_epoch_time
        writer.add_scalar('Total time in h',counter.total_time/3600,iter_idx)
        

if __name__ == '__main__':
    args = parser.parse_args()
    image_args = images_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print_gpu_memory_every_5secs()

    torch.random.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("*****device*****", device)
    


    file_name = "%s_%s_%s" % ("model", "pipeit_navigation", str(args.seed))
    ppo_trained_file_name = "%s_%s_%s" % ("model", "pipeit_navigation", str(args.seed))

    if args.sageMaker:
        model_dir = os.environ['SM_MODEL_DIR']
        tensorBoard_dir = '/opt/ml/output/tensorboard/'
        writer = SummaryWriter(tensorBoard_dir)
    else:
        model_dir = "./trained_models"
        writer = SummaryWriter()

    model = ActorCritic(args.text_input_length,args.depth_map_length,args.action_direction_length).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    normalizedEnv = None#NormalizedEnv(np.zeros(shape=(1,args.text_input_length)).shape,args.parallel_workers,args.depth_map_length)

    if not args.compute_dynamic_stat:
        images_mean,images_std,texts_mean,texts_std = get_data_statistics(Path("data_stat.h5"))
        data_stat ={'images_mean':127.5,'images_std':127.5,'texts_mean':texts_mean,'texts_std':texts_std}


    if args.load_model:
        load_model(_model = model,filename = "%s" % (ppo_trained_file_name), directory="./pytorch_models")
        images_mean,images_std,texts_mean,texts_std = get_data_statistics(Path("data_stat.h5"))
        data_stat ={'images_mean':127.5,'images_std':127.5,'texts_mean':texts_mean,'texts_std':texts_std}
        normalizedEnv.ob_rms.mean = texts_mean
        normalizedEnv.ob_rms.var = np.power(texts_std,2)
        normalizedEnv.ob_rms.count = 200000

    env = None#parallel_envs(args.parallel_workers,args.text_input_length,args.action_direction_length)


    teacher = get_teacher()

    counter = Counter()

    train_ppo(env,model,optimizer,normalizedEnv,writer,teacher,counter,image_args,args,device,ppo_trained_file_name,None,model_dir)

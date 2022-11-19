import time
import numpy as np
import torch
from TrainingObject import *
from utils import check_done
from utils import load_model
import random
from env_ursina import*
from models import ActorCritic
import logging

@torch.no_grad()
def traj_collecting(model,env,teacher,normalizedEnv,counter,writer,device,args,q_model,q_traj,thread_index):

    done_true = np.full((args.envs_per_worker, 1), 1)
    done = np.full((args.envs_per_worker, 1), 0)
    reward = np.zeros(shape=(args.envs_per_worker,1))


    num_episodes = 0

    env_params = []
        
    values_episode = TrainingObject_parallel(args.envs_per_worker)
    terminals_episode = TrainingObject_parallel(args.envs_per_worker)
    texts_inputs_episode = TrainingObject_parallel(args.envs_per_worker)
    actions_masks_episode = TrainingObject_parallel(args.envs_per_worker)
    actions_directions_episode = TrainingObject_parallel(args.envs_per_worker)
    actions_directions_log_probs_episode = TrainingObject_parallel(args.envs_per_worker)
    rewards_episode = TrainingObject_parallel(args.envs_per_worker)


    total_actions_directions_log_probs = []
    total_values = []  
    total_texts_inputs = []
    total_actions_directions = []
    total_action_masks = []
    total_returns = []
    total_advantages = []


    for i in range(args.envs_per_worker):
        params = np.array([random.random() for i in range(2)])#teacher.get_env_params()
        env_params.append(params)

    

    text_input,mask = env.reset(env_params,env.dummy_obs,env.dummy_mask, done_true) 
    text_input,mask = normalizedEnv.reset(text_input,mask,done_true)
    

    while counter.iter_epoch < args.num_steps or 0 in done:
    

        start = time.time()

        text_input = torch.FloatTensor(text_input).to(device)
        mask = torch.BoolTensor(mask).to(device)


        action_direction_dist,value = model(text_input,mask)


        action_direction = action_direction_dist.sample().reshape(-1,1)

  
        next_text_input, next_mask,reward,done = env.step(action_direction.cpu().numpy(),done,reward,text_input.cpu().numpy())

    
        done_bool,_ = check_done(done)

        writer.add_histogram('env rewards without normalization,',reward,counter.iter_index)

        next_text_input,normalized_reward = normalizedEnv.step(next_text_input,reward,done)


        axis_log_prob = action_direction_dist.log_prob(action_direction.flatten()).reshape(-1,1)

        actions_directions_log_probs_episode.append(axis_log_prob,done)
        values_episode.append(value,done)
        terminals_episode.append(torch.FloatTensor(1 - done_bool).to(device),done)        
        texts_inputs_episode.append(text_input,done)
        actions_masks_episode.append(mask,done)
        actions_directions_episode.append(action_direction,done)
        rewards_episode.append(torch.FloatTensor(normalized_reward).to(device),done)

        text_input = next_text_input
        mask = next_mask

        parallel_steps = args.envs_per_worker - np.count_nonzero(done == -1)

        counter.iter_index += parallel_steps
        counter.iter_epoch += parallel_steps
        counter.iter_test += parallel_steps


        end = time.time()
        total_iter_time = end - start
        writer.add_scalar('Iteration time in ms',total_iter_time*1000/parallel_steps,counter.iter_index)
        print(f'time for one iteration is :{total_iter_time*1000/parallel_steps}')  


        if 1 in done:

            start = time.time()

            parallel_steps = np.count_nonzero(done == 1)
            num_episodes += np.count_nonzero(done == 1)


            print(f"number of completed episodes: {num_episodes}")

            next_text_input = torch.FloatTensor(text_input).to(device)
            next_mask = torch.BoolTensor(mask).to(device)

            _,next_value = model(next_text_input,next_mask)             
                
            next_value = values_episode.get_next_value(next_value,done)
            valus = values_episode.get_completed(done)
            termnls = terminals_episode.get_completed(done)
            rewards = rewards_episode.get_completed(done)

            retrn,advantage = values_episode.compute_gae(next_value, rewards, termnls, valus)


            total_values +=valus
            total_returns += retrn
            total_advantages+=advantage
            total_texts_inputs += texts_inputs_episode.get_completed(done)
            total_actions_directions += actions_directions_episode.get_completed(done)
            total_action_masks +=actions_masks_episode.get_completed(done)            
            total_actions_directions_log_probs += actions_directions_log_probs_episode.get_completed(done)


            if counter.iter_epoch <  args.num_steps:
                for i in range(args.envs_per_worker):
                    if done[i] == 1:
                        params = np.array([random.random() for i in range(2)])#teacher.get_env_params()
                        env_params[i] = params


                text_input,mask = env.reset(env_params,text_input,mask,done) 
                text_input,mask = normalizedEnv.reset(text_input,mask,done)



                done = np.full((args.envs_per_worker, 1), 0)


                end = time.time()
                total_iter_time = end - start
                writer.add_scalar('Iteration time in ms',total_iter_time*1000/parallel_steps,counter.iter_index)
                print(f'time for one iteration is :{total_iter_time*1000/parallel_steps}')  
            else:
                for i in range(done.shape[0]):
                    if done[i] == 1:
                        done[i] = -1


        print(f'**[ITERATION NUMBER: {counter.iter_index}]**')
        print(f'epoch_steps number: {counter.iter_epoch}')
        #print(f"Thread {thread_index} {done}")          
    

    print(f"FINISH COLLECTING TRAJ")

    env.clear()

    traj = {
            "actions_directions_log_probs":total_actions_directions_log_probs,
            "values":total_values,"text_inputs":total_texts_inputs,
            "actions_directions":total_actions_directions,"actions_mask":total_action_masks
            ,"returns": total_returns, "advantages": total_advantages}
    
    q_traj.put(traj)


@torch.no_grad()
def get_epoch_trajectories_ursina(model,env,teacher,normalizedEnv,counter,writer,device,args,q_model,q_traj,thread_index):

    if model == None:
        model = ActorCritic(args.text_input_length,args.depth_map_length,args.action_direction_length).to(device)
        
    while True:
        
        model_dic = q_model.get()
        model.load_state_dict(model_dic)

        traj_collecting(model,env,teacher,normalizedEnv,counter,writer,device,args,q_model,q_traj,thread_index)

    

@torch.no_grad()
def test_ursina(model,env,teacher,normalizedEnv,counter,args,image_args,device):

    print(f"[ENTER TEST LOOP]")

    env.set_active_workers(args.parallel_workers_test)

    print("ACTIVE WORKERS WAS CHANGED")

    done = np.full((args.parallel_workers_test, 1), 0)
    done_true = np.full((args.parallel_workers_test, 1), 1)
    reward = np.zeros(shape=(args.parallel_workers_test,1))
    total_reward = np.zeros(shape=(args.parallel_workers_test,1))
    total_steps = np.zeros(shape=(args.parallel_workers_test,1))

    env_params = []

    for i in range(args.parallel_workers_test):
        if teacher:
            params = teacher.get_env_params()
        else:
            params = np.array([random.random() for i in range(2)])
        env_params.append(params)

    text_input,mask = env.reset(env_params,env.dummy_obs,env.dummy_mask,done_true)
    text_input,mask = normalizedEnv.reset(text_input,mask,done_true,True)

    while 0 in done:       

        text_input = torch.FloatTensor(text_input).to(device)
        mask = torch.BoolTensor(mask).to(device)

        action_direction_dist,_ = model(text_input,mask)

        action_direction = action_direction_dist.sample().reshape(-1,1)


        next_text_input, next_mask,reward,done = env.step(action_direction.cpu().numpy(),done,reward,text_input.cpu().numpy())

        check_done(done)

        next_text_input,reward = normalizedEnv.step(next_text_input,reward,done, True)

        print(done)

        for i in range(args.parallel_workers_test):
            if done[i] == 0:
                total_steps[i] += 1
                total_reward[i] += reward[i]
            elif done[i] == 1:
                total_steps[i] +=1
                total_reward[i] += reward[i]
                done[i] = -1

        print(total_reward)

        text_input = next_text_input
        mask = next_mask

    print("total steps", total_steps)
    print("total reward",total_reward)

    counter.test_average_reward = total_reward.sum()/args.parallel_workers_test
    counter.test_average_steps = total_steps.sum()/args.parallel_workers_test

    env.clear()
    print("[EXIT TEST LOOP]")


def ppo_update(device,ppo_epochs, mini_batch_size, text_inputs,actions_direction,actions_directions_log_probs, returns,advantages,action_masks,values,counter,optimizer,model,writer,args ,clip_param=0.1, max_grad_norm = 0.5):
    
    batch_size = text_inputs.size(0)
    total_iter = (batch_size // mini_batch_size)

    for j in range(ppo_epochs):
        print(f"ppo epoch {j} out of {ppo_epochs}")
        counter.iter_ppo +=1
        total_loss = 0
        total_critic_loss = 0
        total_actor_loss = 0
        total_actions_direction_entropy = 0


        for text_inputs_,actions_directions_,old_actions_directions_log_probs,returns_, advantages_,action_masks_,values_ in ppo_iter(mini_batch_size, text_inputs,actions_direction ,actions_directions_log_probs, returns,advantages,action_masks,values):      
                        
            returns_ = returns_.to(device)
            values_ = values_.to(device)
            advantages_ = advantages_.to(device)
            
            new_actions_directions_dist,new_values = model(text_inputs_.to(device),action_masks_.to(device))

            actions_directions_entropy = new_actions_directions_dist.entropy().mean()
            new_actions_directions_log_probs = new_actions_directions_dist.log_prob(actions_directions_.flatten().to(device)).reshape(-1,1)

            new_log_probs = new_actions_directions_log_probs
            old_log_probs = old_actions_directions_log_probs
            
            ratio = (new_log_probs - old_log_probs.to(device)).exp()
            surr1 = ratio * advantages_
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_
        
            actor_loss = -torch.min(surr1,surr2).mean()
                    
            v_loss_unclipped = (new_values - returns_).pow(2)
            v_clipped = values_ + torch.clamp(
                    new_values - values_,
                    -clip_param,
                    clip_param,
                )
            v_loss_clipped = (v_clipped - returns_).pow(2)
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)

            critic_loss = args.critic_loss_coeff * v_loss_max.mean() 
            

            loss = critic_loss + actor_loss - args.entropy_coeff* actions_directions_entropy

            optimizer.zero_grad()  
            with torch.autograd.set_detect_anomaly(True):   
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_actions_direction_entropy += (actions_directions_entropy*args.entropy_coeff).item()

                        

        writer.add_scalar('Total loss', total_loss/total_iter,counter.iter_ppo)
        writer.add_scalar('Critic loss', total_critic_loss/total_iter,counter.iter_ppo)
        writer.add_scalar('Actor loss', total_actor_loss/total_iter,counter.iter_ppo)
        writer.add_scalar('Actions directions entropy', total_actions_direction_entropy/total_iter,counter.iter_ppo)
        

def to_torch_tensor(list, dim = 0):
    total_list =  [item for sublist in list for item in sublist]
    return torch.stack(total_list,dim)

def ppo_iter(mini_batch_size, text_inputs,actions_direction,actions_directions_log_probs, returns,advantages,action_masks,values):
    batch_size = text_inputs.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield text_inputs[rand_ids, :],actions_direction[rand_ids, :],actions_directions_log_probs[rand_ids, :] ,returns[rand_ids, :], advantages[rand_ids, :],action_masks[rand_ids, :],values[rand_ids, :]

def collector(q_traj):
    traj = {
            "actions_directions_log_probs":[],
            "values":[],"text_inputs":[],
            "actions_directions":[],"actions_mask":[],
            "returns": [], "advantages": []
            }

    while True:
        time.sleep(1)
        print("FINISHED THREADS", q_traj.qsize())
        if q_traj.full():
            print("is full", q_traj.full())
            break
    
    while not q_traj.empty():
        t = q_traj.get()
        traj["text_inputs"] += t["text_inputs"]
        traj["actions_directions"] += t["actions_directions"]
        traj["actions_directions_log_probs"] += t["actions_directions_log_probs"]
        traj["actions_mask"] += t["actions_mask"]
        traj["values"] += t["values"]
        traj["returns"] +=t["returns"]
        traj["advantages"] += t["advantages"]

    return traj
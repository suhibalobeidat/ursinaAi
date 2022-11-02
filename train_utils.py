import time
import numpy as np
import torch
from TrainingObject import *
from utils import check_done
from utils import load_model
import random
from env_ursina import*
from models import ActorCritic


@torch.no_grad()
def get_epoch_trajectories_ursina(model,env,teacher,normalizedEnv,counter,writer,device,args,q_model,q_traj):

    if model == None:
        model = ActorCritic(args.text_input_length,args.depth_map_length,args.action_direction_length).to(device)
        
    while True:

        model_dic = q_model.get()

        model.load_state_dict(model_dic)

        done_true = np.full((args.envs_per_worker, 1), 1)
        done = np.full((args.envs_per_worker, 1), 0)
        reward = np.zeros(shape=(args.envs_per_worker,1))

        num_episodes = 0

        env_params = []
            
        #env.set_active_workers(args.parallel_workers)

        values_episode = TrainingObject_parallel(args.envs_per_worker)
        terminals_episode = TrainingObject_parallel(args.envs_per_worker)
        texts_inputs_episode = TrainingObject_parallel(args.envs_per_worker)
        actions_masks_episode = TrainingObject_parallel(args.envs_per_worker)
        actions_length_episode = TrainingObject_parallel(args.envs_per_worker)
        actions_directions_episode = TrainingObject_parallel(args.envs_per_worker)
        actions_length_log_probs_episode = TrainingObject_parallel(args.envs_per_worker)
        actions_directions_log_probs_episode = TrainingObject_parallel(args.envs_per_worker)
        rewards_episode = TrainingObject_parallel(args.envs_per_worker)
        hidden_states_episode = TrainingObject_parallel(args.envs_per_worker, False)
        cell_states_episode = TrainingObject_parallel(args.envs_per_worker,False)

        total_actions_length_log_probs = []
        total_actions_directions_log_probs = []
        total_values = []  
        total_texts_inputs = []
        total_actions_length = []
        total_actions_directions = []
        total_terminals = []   
        total_action_masks = []
        total_next_values = []
        total_rewards   = []
        total_returns = []
        total_advantages = []
        total_hidden_states = []
        total_cell_states = []

        for i in range(args.envs_per_worker):
            params = teacher.get_env_params()
            env_params.append(params)

        

        text_input,mask = env.reset(env_params,env.dummy_obs,env.dummy_mask, done_true) 
        text_input,mask = normalizedEnv.reset(text_input,mask,done_true)

        model.get_init_state(args.envs_per_worker,device)
        
        while counter.iter_epoch < args.num_steps or 0 in done:#done.sum() != args.parallel_workers:
            start = time.time()

            #inference_start_time = time.time()

            hidden_state = model.hidden_cell[0]
            cell_state = model.hidden_cell[1]

            writer.add_histogram('text,',text_input,counter.iter_index)

            text_input = torch.FloatTensor(text_input).to(device)
            mask = torch.BoolTensor(mask).to(device)

        
            if normalizedEnv.ob_rms:
                writer.add_histogram('text_mean,',normalizedEnv.ob_rms.mean,counter.iter_index)
                writer.add_histogram('text_std,',np.sqrt(normalizedEnv.ob_rms.var),counter.iter_index)


            action_length_dist,action_direction_dist,value = model(text_input,mask)

            action_direction = action_direction_dist.sample().reshape(-1,1)
            action_length = action_length_dist.sample().expand_as(action_direction)

            action = np.concatenate((action_length.cpu().numpy()*10,action_direction.cpu().numpy()), axis=-1)

            next_text_input, next_mask,reward,done = env.step(action,done,reward,text_input.cpu().numpy())
            
            #print(done)

            done_bool,successful_ep = check_done(done)

            #inference_end_time = time.time()
            #writer.add_scalar('time for inference in ms',(inference_end_time - inference_start_time)*1000,counter.iter_index)


            #reward_start_time = time.time()

            #print("reward without normalization", reward)
            writer.add_histogram('env rewards without normalization,',reward,counter.iter_index)
            next_text_input,normalized_reward = normalizedEnv.step(next_text_input,reward,done)
            #print("reward with normalization", normalized_reward)

            writer.add_histogram('env rewards with normalization,',normalized_reward,counter.iter_index)

            #reward_end_time = time.time()

            #print(f'time for reward adding:{(reward_end_time - reward_start_time)*1000}') 
            #writer.add_scalar('time for reward adding in ms',(reward_end_time - reward_start_time)*1000,counter.iter_index)

            #storing_data_start_time = time.time()
            action_log_prob = action_length_dist.log_prob(action_length)            
            axis_log_prob = action_direction_dist.log_prob(action_direction.flatten()).reshape(-1,1)


            actions_length_log_probs_episode.append(action_log_prob.cpu(),done)
            actions_directions_log_probs_episode.append(axis_log_prob.cpu(),done)
            values_episode.append(value.cpu(),done)
            terminals_episode.append(torch.FloatTensor(1 - done_bool).cpu(),done)        
            texts_inputs_episode.append(text_input.cpu(),done)
            actions_masks_episode.append(mask.cpu(),done)
            actions_length_episode.append(action_length.cpu(),done)
            actions_directions_episode.append(action_direction.cpu(),done)
            rewards_episode.append(torch.FloatTensor(normalized_reward).cpu(),done)
            hidden_states_episode.append(hidden_state.cpu(),done)
            cell_states_episode.append(cell_state.cpu(),done)
    
            text_input = next_text_input
            mask = next_mask

            counter.iter_index += args.envs_per_worker - np.count_nonzero(done == -1)
            counter.iter_epoch += args.envs_per_worker - np.count_nonzero(done == -1)
            counter.iter_test += args.envs_per_worker - np.count_nonzero(done == -1)

            #storing_data_end_time = time.time()
            #writer.add_scalar('time for storing data in ms',(storing_data_end_time - storing_data_start_time)*1000,counter.iter_index)

            if 1 in done:
                #inside_start_time = time.time()

                num_episodes += np.count_nonzero(done == 1)

                if successful_ep.sum() > 0:
                    counter.iter_successful_ep += successful_ep.sum()

                print(f"number of completed episodes: {num_episodes}")

                next_text_input = torch.FloatTensor(text_input).to(device)
                next_mask = torch.BoolTensor(mask).to(device)
    

                _,_,next_value = model(next_text_input,next_mask)             
                    
                next_value = values_episode.get_next_value(next_value.cpu(),done)
                valus = values_episode.get_completed(done)
                termnls = terminals_episode.get_completed(done)
                rewards = rewards_episode.get_completed(done)

                retrn,advantage = values_episode.compute_gae(next_value, rewards, termnls, valus)


                

                total_terminals +=termnls
                total_values +=valus
                total_next_values += next_value
                total_rewards += rewards
                total_returns += retrn
                total_advantages+=advantage
                total_texts_inputs += texts_inputs_episode.get_completed(done)
                total_actions_length += actions_length_episode.get_completed(done)
                total_actions_directions += actions_directions_episode.get_completed(done)
                total_action_masks +=actions_masks_episode.get_completed(done)            
                total_actions_length_log_probs +=actions_length_log_probs_episode.get_completed(done)
                total_actions_directions_log_probs += actions_directions_log_probs_episode.get_completed(done)
                total_hidden_states += hidden_states_episode.get_completed(done)
                total_cell_states += cell_states_episode.get_completed(done)

                r_index = 0
                for i in range(done.shape[0]):
                    if done[i] == 1:
                        #writer.add_scalar("distance",env_params[i][0],counter.iter_index)
                        writer.add_scalar("width",env_params[i][0],counter.iter_index)
                        writer.add_scalar("height",env_params[i][1],counter.iter_index)
                        teacher.record_train_episode(retrn[r_index][0], counter.iter_index,env_params[i])
                        r_index+=1

                if counter.iter_epoch <  args.num_steps:
                    for i in range(args.envs_per_worker):
                        if done[i] == 1:
                            params = teacher.get_env_params()
                            env_params[i] = params


                    text_input,mask = env.reset(env_params,text_input,mask,done) 
                    text_input,mask = normalizedEnv.reset(text_input,mask,done)

                    done = np.full((args.envs_per_worker, 1), 0)

                    model.get_init_state(args.envs_per_worker,device)
                else:
                    for i in range(done.shape[0]):
                        if done[i] == 1:
                            done[i] = -1

                #inside_end_time = time.time()
                #writer.add_scalar('time for inside in ms',(inside_end_time - inside_start_time)*1000,counter.iter_index)

            print(f'**[ITERATION NUMBER: {counter.iter_index}]**')
            print(f'epoch_steps number: {counter.iter_epoch}')
        
                
            end = time.time()
            total_iter_time = end - start
            writer.add_scalar('Iteration time in ms',total_iter_time*1000/args.envs_per_worker,counter.iter_index)
            print(f'time for one iteration is :{total_iter_time*1000/args.envs_per_worker}')      

        print(f"FINISH COLLECTING TRAJ")

        env.clear()

        traj = {"actions_length_log_probs":total_actions_length_log_probs,
                "actions_directions_log_probs":total_actions_directions_log_probs,
                "values":total_values,"next_values":total_next_values,"text_inputs":total_texts_inputs,
                "actions_length":total_actions_length,
                "actions_directions":total_actions_directions,"actions_mask":total_action_masks
                ,"terminals": total_terminals,"returns": total_returns, "advantages": total_advantages
                ,"rewards":total_rewards,"hidden_states": total_hidden_states
                , "cell_states": total_cell_states}
        
        q_traj.put(traj)

        #return traj

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
    #return
    text_input,mask = normalizedEnv.reset(text_input,mask,done_true,True)

    model.get_init_state(args.parallel_workers_test,device)

    while 0 in done:#done.sum() != args.parallel_workers_test:        

        text_input = torch.FloatTensor(text_input).to(device)
        mask = torch.BoolTensor(mask).to(device)

        action_length_dist,action_direction_dist,_ = model(text_input,mask)

        action_direction = action_direction_dist.sample().reshape(-1,1)
        action_length = action_length_dist.sample().expand_as(action_direction)

        #print("state",text_input[:,361:])
        #print("mask",mask)
        #print("action", action_direction)
        action = np.concatenate((action_length.cpu().numpy()*10,action_direction.cpu().numpy()), axis=-1)
        
        #start_time = time.time()
        next_text_input, next_mask,reward,done = env.step(action,done,reward,text_input.cpu().numpy())

        #end_time = time.time()
        #total_time = (end_time-start_time)*1000
        #print(total_time/args.parallel_workers_test)
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


def ppo_update(device,gamma,ppo_epochs, mini_batch_size, text_inputs,actions_length,actions_direction,actions_length_log_probs ,actions_directions_log_probs, returns,advantages,action_masks,values,hidden_states,cell_states,counter,optimizer,model,writer,args ,clip_param=0.1, max_grad_norm = 0.5):
    

    batch_size = text_inputs.size(0)
    #scaler = torch.cuda.amp.GradScaler()
    batch_counter = 0
    for j in range(ppo_epochs):
        print(f"ppo epoch {j} out of {ppo_epochs}")
        counter.iter_ppo +=1
        total_loss = 0
        total_critic_loss = 0
        total_actor_loss = 0
        total_actions_length_entropy = 0
        total_actions_direction_entropy = 0
        total_bc_loss = 0


        for text_inputs_, actions_length_,actions_directions_,old_actions_length_log_probs,old_actions_directions_log_probs,returns_, advantages_,action_masks_,values_,hidden_states_,cell_states_ in ppo_iter(mini_batch_size, text_inputs,actions_length,actions_direction,actions_length_log_probs ,actions_directions_log_probs, returns,advantages,action_masks,values,hidden_states,cell_states):      

            optimizer.zero_grad()
            #with torch.cuda.amp.autocast():
            returns_ = returns_.to(device)
            values_ = values_.to(device)
            advantages_ = advantages_.to(device)
            #hidden_states_ = hidden_states_.to(device)
            #cell_states_ = cell_states_.to(device)
            
            new_actions_length_dist,new_actions_directions_dist,new_values = model(text_inputs_.to(device),action_masks_.to(device),hidden_cell = [hidden_states_,cell_states_])
            #actions_length_entropy = new_actions_length_dist.entropy().mean()
            #new_actions_length_log_probs = new_actions_length_dist.log_prob(actions_length_.to(device))

            actions_directions_entropy = new_actions_directions_dist.entropy().mean()
            new_actions_directions_log_probs = new_actions_directions_dist.log_prob(actions_directions_.flatten().to(device)).reshape(-1,1)

            #new_log_probs = new_actions_length_log_probs + new_actions_directions_log_probs
            #old_log_probs = old_actions_length_log_probs + old_actions_directions_log_probs
            new_log_probs = new_actions_directions_log_probs
            old_log_probs = old_actions_directions_log_probs
            
            ratio = (new_log_probs - old_log_probs.to(device)).exp()
            surr1 = ratio * advantages_
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_
        
            actor_loss = -torch.min(surr1,surr2).mean()
                    
            v_loss_unclipped = (new_values - returns_) ** 2
            v_clipped = values_ + torch.clamp(
                    new_values - values_,
                    -clip_param,
                    clip_param,
                )
            v_loss_clipped = (v_clipped - returns_) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)

            critic_loss = args.critic_loss_coeff * v_loss_max.mean() 
            

            loss = critic_loss + actor_loss- args.entropy_coeff* actions_directions_entropy#-args.entropy_coeff* actions_length_entropy #+0.01*images_loss +0.01*visible_spaces_loss #+ 0.00001*spaces_height_loss

            
            """ scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update() """

            loss.backward()

            #batch_counter +=1
            #if batch_counter * 128 == mini_batch_size:
            #batch_counter = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            #total_bc_loss += float(gamma * bc_loss)
            #total_actions_length_entropy += (actions_length_entropy*args.entropy_coeff).item()
            total_actions_direction_entropy += (actions_directions_entropy*args.entropy_coeff).item()

                        
        total_iter = (batch_size // mini_batch_size)

        writer.add_scalar('Total loss', total_loss/total_iter,counter.iter_ppo)
        #writer.add_scalar('bs loss loss', total_bc_loss/total_iter,iter_idx)
        writer.add_scalar('Critic loss', total_critic_loss/total_iter,counter.iter_ppo)
        writer.add_scalar('Actor loss', total_actor_loss/total_iter,counter.iter_ppo)
        #writer.add_scalar('Actions length entropy', total_actions_length_entropy/total_iter,iter_idx)
        writer.add_scalar('Actions directions entropy', total_actions_direction_entropy/total_iter,counter.iter_ppo)
        

def to_torch_tensor(list, dim = 0):
    total_list =  [item for sublist in list for item in sublist]
    return torch.stack(total_list,dim)

def ppo_iter(mini_batch_size, text_inputs,actions_length,actions_direction,actions_length_log_probs ,actions_directions_log_probs, returns,advantages,action_masks,values,hidden_states, cell_states):
    batch_size = text_inputs.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield text_inputs[rand_ids, :], actions_length[rand_ids, :],actions_direction[rand_ids, :] ,actions_length_log_probs[rand_ids, :],actions_directions_log_probs[rand_ids, :] ,returns[rand_ids, :], advantages[rand_ids, :],action_masks[rand_ids, :],values[rand_ids, :],hidden_states[:,rand_ids, :],cell_states[:,rand_ids, :]

def collector(q_traj):
    traj = {"actions_length_log_probs":[],
            "actions_directions_log_probs":[],
            "values":[],"next_values":[],"text_inputs":[],
            "actions_length":[],
            "actions_directions":[],"actions_mask":[]
            ,"terminals": [],"returns": [], "advantages": []
            ,"rewards":[],"hidden_states": []
            , "cell_states": []}

    while True:
        time.sleep(1)
        if q_traj.full():
            print("is full", q_traj.full())
            break
    
    while not q_traj.empty():
        t = q_traj.get()
        traj["text_inputs"] += t["text_inputs"]
        traj["actions_length"] += t["actions_length"]
        traj["actions_directions"] += t["actions_directions"]
        traj["actions_length_log_probs"] += t["actions_length_log_probs"]
        traj["actions_directions_log_probs"] += t["actions_directions_log_probs"]
        traj["actions_mask"] += t["actions_mask"]
        traj["values"] += t["values"]
        traj["next_values"] += t["next_values"]
        traj["terminals"] += t["terminals"]
        traj["hidden_states"] +=t["hidden_states"]
        traj["cell_states"] +=t["cell_states"]
        traj["rewards"] +=t["rewards"]
        traj["returns"] +=t["returns"]
        traj["advantages"] += t["advantages"]

    return traj
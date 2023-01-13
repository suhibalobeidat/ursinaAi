import threading
from enums import ConnectorShape, Commands,ActionSpace
from room_layout import Layout
from mep_curve_system import System
from multiprocessing import Queue
from transform import Transform
from utils_ursina import *
import random
from multiprocessing import Process,Queue
from ursina import *


def create_env(min_size,max_size,receive_queue = None,send_queue = None, same_process = False):
    app = Ursina()
    navigation_env = Navigation_env(min_size,max_size,receive_queue,send_queue)
    app.set_env(navigation_env)

    if not same_process:
        app.run()
    else:
        return app

class Navigation_env():
    def __init__(self,min_size,max_size,receive_queue,send_queue):
        self.action_space = ActionSpace.DISCRETE
        self.receive_queue = receive_queue
        self.send_queue = send_queue
        self.radius_mul = 1
        self.max_iteration = 100
        self.max_room_number = 5
        self.actions = 29
        self.min_duct_size = min_size#mm
        self.max_duct_size = max_size#mm
        self.min_distance = 10#ft
        self.max_distance = 100#ft
        self.system = None
        self.layout = None
        self.done = True

    def init(self):
        self.system = System(shape=ConnectorShape.Rectangle,max_iteration=self.max_iteration,radius_mul=self.radius_mul)
        self.layout = Layout(max_rooms_number=self.max_room_number, min_distance=self.min_distance)

        self.send_queue.put(1)


    def reset(self, goal):
        #distance = correct_value(goal[0],0,1,self.min_distance,self.max_distance)
        
        #goal_point = get_point_at_distance(distance)

        if not self.layout.is_last_room:
            self.system.create_detection_system()

            total_status = []

            total_status.extend(self.system.get_status(self.layout.next_rect))
            total_status.extend(self.system.get_action_mask())
            self.done = False
            self.system.is_done = False
            self.system.iteration = 0

            self.send_queue.put(total_status)

            return

        self.system.set_dim(correct_value(goal[1],0,1,self.min_duct_size,self.max_duct_size),correct_value(goal[2],0,1,self.min_duct_size,self.max_duct_size))

        self.system.reset()
        self.layout.reset()

        self.layout.generate_layout(self.system.system_segmants[-1],goal[3])
        self.layout.get_path_rects()

        #goal_point = self.layout.get_goal_point(distance)
        #goal_point = self.generate_random_point(distance)

        

        #self.system.set_goal(goal_point)
        self.system.create_detection_system()

        total_status = []

        total_status.extend(self.system.get_status(self.layout.next_rect))
        total_status.extend(self.system.get_action_mask())
        self.done = False

        self.send_queue.put(total_status)

    def clear(self):
        self.system.reset()
        self.layout.reset()

        self.send_queue.put([True])

    def step(self,action):
        if self.action_space == ActionSpace.DISCRETE:
            if self.system.create_segmant(0,action[0]):
                self.system.create_detection_system()   
        else: 
            if self.system.create_segmant(action[0],action[1]):
                self.system.create_detection_system()

        next_rect,is_new_room = self.layout.get_next_rect(self.system.system_segmants[-1])
    
        total_status = []
        total_status.extend(self.system.get_status(next_rect))

        
        if self.system.check_for_collision():
            self.system.is_done = 1
            self.done = True
            self.layout.is_last_room = True
        elif self.system.is_successful(is_new_room):
            self.system.is_done = 3
            self.done = True
        elif self.system.max_iteration_exceeded() or self.layout.is_last_room: 
            self.system.is_done = 2
            self.done = True
            self.layout.is_last_room = True
        else:
            self.system.is_done = 0
            self.done = False


        total_status.extend(self.system.get_action_mask())
        total_status.append(int(self.system.is_done ))
        total_status.append(self.system.get_reward(is_new_room))

        self.send_queue.put(total_status)

    def get_random_action(self):
        actions = np.arange(self.actions)
        actions_mask = np.array(self.system.action_mask,dtype=np.bool)
        masked_actions = actions[actions_mask]
        random_action_direction  = masked_actions[random.randint(0,len(masked_actions)-1)]
        random_action_length = random.random()*10
        self.send_queue.put([random_action_length,random_action_direction])


    def update(self):

        if self.receive_queue.empty():
            return
        
        data = self.receive_queue.get()

        command = int(data[0])
        if len(data) > 1:
            remaining_data = data[1:]

        #print("command",command)

        if Commands(command) == Commands.init:
            self.init()
        elif Commands(command) == Commands.reset:
            self.reset(remaining_data)
        elif Commands(command) == Commands.setp:
            self.step(remaining_data)
        elif Commands(command) == Commands.get_action:
            self.get_random_action()
        elif Commands(command) == Commands.done:
            self.send_queue.put([self.done])   
        elif Commands(command) == Commands.clear:
            self.clear()


class MyQueue():
    def __init__(self):
        self.queue = []

    def put(self,data):
        self.queue.append(data)

    def get(self):
        data = self.queue.pop(0)
        return data

    def empty(self):
        if len(self.queue) == 0:
            return True
        else:
            return False
        

class env_interface():
    def __init__(self,min_size,max_size,number = 0, same_process = False):
        self.same_process = same_process
        self.number = number

        if same_process:
            self.receive_queue = MyQueue()
            self.send_queue = MyQueue()
            self.app = self.process = create_env(min_size,max_size,self.send_queue,self.receive_queue, same_process)
        else:
            self.receive_queue = Queue() 
            self.send_queue = Queue() 
            self.process = Process(target=create_env, args=(min_size,max_size,self.send_queue,self.receive_queue,same_process))
            self.process.start()
        

    def init(self):
        data = []
        data.append(0.)
        self.send_queue.put(data)

        if self.same_process:
            self.app.step()
        
        received_data = self.receive_queue.get()

        return received_data

    def reset(self,goal):

        data = []
        data.append(1.)
        data.append(20.)
        data.extend(goal)

        self.send_queue.put(data)
        if self.same_process:
            self.app.step()
        received_data = self.receive_queue.get()

        return received_data

    def step(self,action):

        data = []
        data.append(2.)

        if isinstance(action,list):
            data.extend(action)
        else:
            data.append(action)
            data.append(0)

        self.send_queue.put(data)
        if self.same_process:
            self.app.step()
        received_data = self.receive_queue.get()

        return received_data

    def get_random_action(self):
        data = []
        data.append(3.)
        self.send_queue.put(data)
        if self.same_process:
            self.app.step()
        received_data = self.receive_queue.get()
        return received_data

    def is_done(self):
        data = []
        data.append(4.)
        self.send_queue.put(data)
        if self.same_process:
            self.app.step()
        received_data = self.receive_queue.get() 
        return received_data[0]

    def clear(self):
        data = []
        data.append(5.)
        self.send_queue.put(data)
        if self.same_process:
            self.app.step()
        received_data = self.receive_queue.get()
        return received_data


class parallel_envs():
    def __init__(self,parallel_envs_count, obs_size, mask_size):
        self.parallel_envs_count = parallel_envs_count
        self.envs = []
        self.mask_size = mask_size
        self.obs_size = obs_size
        self.init()
        self.dummy_obs,self.dummy_mask = self.get_dummy_init_state(parallel_envs_count)
        self.active_workers = parallel_envs_count

    def set_active_workers(self,active_workers):
        self.active_workers = active_workers
        self.dummy_obs,self.dummy_mask = self.get_dummy_init_state(active_workers)

    def init(self):
        threads = []
        for i in range(self.parallel_envs_count):
            env = env_interface(75,250,i)
            self.envs.append(env)  
            threads.append(threading.Thread(target=env.init))   

        for thread in threads:
            thread.start()

    def reset_thread(self,i,goal):
        receivedData = self.envs[i].reset(goal)

        new_obs = receivedData[:len(receivedData)-self.mask_size]
        new_mask = receivedData[len(receivedData)-self.mask_size:]

        self.new_obs[i] = new_obs
        self.mask[i] = new_mask

    def step_thread(self,i,action):
        receivedData = self.envs[i].step(action)

        new_obs = receivedData[:len(receivedData)-(self.mask_size+2)]
        new_mask = receivedData[len(receivedData)-(self.mask_size+2):len(receivedData)-2]
        done = receivedData[len(receivedData)-2]
        reward = receivedData[len(receivedData)-1]

        self.new_obs[i] = new_obs
        self.mask[i] = new_mask
        self.reward[i] = reward
        self.done[i] = done

    def clear_thread(self,i):
        receivedData = self.envs[i].clear()

    def get_dummy_init_state(self,active_workers):
        obs = np.zeros(shape=(active_workers,self.obs_size))
        mask = np.zeros(shape=(active_workers,self.mask_size))

        return obs, mask


    def reset(self,goal,obs, mask, done):

        self.new_obs = obs
        self.mask = mask

        threads = []

        for i in range(self.active_workers):
            if self.envs[i].is_done():
                thread = threading.Thread(target=self.reset_thread, args=(i,goal[i].tolist()))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        return self.new_obs,self.mask  

    def step(self,action,done,reward,obs):

        self.new_obs = obs
        self.mask = np.zeros(shape=(self.active_workers,self.mask_size))
        self.done = done
        self.reward = deepcopy(reward)

        threads = []

        for i in range(self.active_workers):
            if not self.envs[i].is_done():
                thread = threading.Thread(target=self.step_thread, args=(i,action[i]))
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

        #print("reward inside env", self.reward)

        return self.new_obs,self.mask,self.reward,self.done

    def get_random_action(self):
        actions = np.zeros(shape=(len(self.envs),2))

        for i in range(len(self.envs)):
            action = self.envs[i].get_random_action()
            actions[i] = action

        return actions

    def clear(self):
        threads = []

        for i in range(self.active_workers):
            thread = threading.Thread(target=self.clear_thread, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def terminate(self):
        for env in self.envs:
            env.process.terminate()

class mul_parallel_envs():
    def __init__(self,envs,parallel_envs_count, obs_size, mask_size):
        self.parallel_envs_count = parallel_envs_count
        self.envs = []
        self.mask_size = mask_size
        self.obs_size = obs_size
        self.init(envs)
        self.dummy_obs,self.dummy_mask = self.get_dummy_init_state(parallel_envs_count)
        self.active_workers = parallel_envs_count

    def set_active_workers(self,active_workers):
        self.active_workers = active_workers
        self.dummy_obs,self.dummy_mask = self.get_dummy_init_state(active_workers)

    def init(self,envs):
        for i in range(len(envs)):
            self.envs.extend(envs[i].envs)

    def reset_thread(self,i,goal):
        receivedData = self.envs[i].reset(goal)

        new_obs = receivedData[:len(receivedData)-self.mask_size]
        new_mask = receivedData[len(receivedData)-self.mask_size:]

        self.new_obs[i] = new_obs
        self.mask[i] = new_mask

    def step_thread(self,i,action):
        receivedData = self.envs[i].step(action)

        new_obs = receivedData[:len(receivedData)-(self.mask_size+2)]
        new_mask = receivedData[len(receivedData)-(self.mask_size+2):len(receivedData)-2]
        done = receivedData[len(receivedData)-2]
        reward = receivedData[len(receivedData)-1]

        self.new_obs[i] = new_obs
        self.mask[i] = new_mask
        self.reward[i] = reward
        self.done[i] = done

    def clear_thread(self,i):
        receivedData = self.envs[i].clear()

    def get_dummy_init_state(self,active_workers):
        obs = np.zeros(shape=(active_workers,self.obs_size))
        mask = np.zeros(shape=(active_workers,self.mask_size))

        return obs, mask


    def reset(self,goal,obs, mask, done):

        self.new_obs = obs
        self.mask = mask

        threads = []

        for i in range(self.active_workers):
            if self.envs[i].is_done():
                thread = threading.Thread(target=self.reset_thread, args=(i,goal[i].tolist()))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        return self.new_obs,self.mask  

    def step(self,action,done,reward,obs):

        self.new_obs = obs
        self.mask = np.zeros(shape=(self.active_workers,self.mask_size))
        self.done = done
        self.reward = deepcopy(reward)

        threads = []

        for i in range(self.active_workers):
            if not self.envs[i].is_done():
                thread = threading.Thread(target=self.step_thread, args=(i,action[i]))
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

        #print("reward inside env", self.reward)

        return self.new_obs,self.mask,self.reward,self.done

    def get_random_action(self):
        actions = np.zeros(shape=(len(self.envs),2))

        for i in range(len(self.envs)):
            action = self.envs[i].get_random_action()
            actions[i] = action

        return actions

    def clear(self):
        threads = []

        for i in range(self.active_workers):
            thread = threading.Thread(target=self.clear_thread, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def terminate(self):
        for env in self.envs:
            env.process.terminate()
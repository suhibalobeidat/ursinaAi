import socket
import numpy as np
import torch
import h5py
from torch.distributions.categorical import Categorical
import os

from models import rlib_model


class Navigator:
    def __init__(self, input_size,action_length,model_path,obs_mean,obs_var):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = load_model("model",model_path,self.device).to(self.device)

        self.obs_mean = np.array(obs_mean)
        self.obs_var = np.array(obs_var)

        self.action_length = action_length
        self.clipob = 10
        self.epsilon = 1e-4
        self.input_size = input_size

        self.host = "127.0.0.1"
        self.port = 65212
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host,self.port))  

        self.terminate = False
        
    def connect(self):

        self.sock.listen()
        self.connection, self.id = self.sock.accept()

        print("CONNECTED!")

    def close(self):
        self.connection.shutdown(socket.SHUT_RDWR) 
        self.connection.close()
    
    def get_obs(self,observation):

        action_mask = torch.FloatTensor(np.array(observation[:self.action_length])).to(self.device)
        obs = np.array(observation[self.action_length:])
        obs = torch.FloatTensor(np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon), -self.clipob, self.clipob)).unsqueeze(dim=0).to(self.device)
        observation = {"obs":obs,"action_mask":action_mask}

        return observation

    def step(self):
        print("INSIDE STEP!!")
        try:
            receivedData = self.connection.recv(9000).decode("UTF-8") #receiveing data in Byte fron C#, and converting it to String
            receivedData = receivedData.split(",")
            receivedData = [float(i) for i in receivedData]
            print("length of received data: ", len(receivedData))
        except:
            #self.close()
            #self.terminate = True
            return

        if len(receivedData) == 1:
            #self.close()
            #self.terminate = True
            return

        obs = self.get_obs(receivedData)
        input_dict = {"obs": obs, "obs_flat":None}
        with torch.no_grad():
            logits,state = self.policy(input_dict,None,None)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample().item()
            value = self.policy.value_function().item()

        print(obs["obs"].shape)
        print("action",action)
        data = [action,value]
        data = ','.join(map(str, data)) #Converting List to a string, example "0,0,0"
        self.connection.sendall(data.encode("UTF-8")) #Converting string to Byte, and sending it to C#


def load_model(filename, directory,device):
    _model = torch.load('%s/%s.pt' % (directory, filename),map_location=device)
    return _model
    

def get_data_statistics(dir,file_name):
    file = h5py.File(dir+"/"+file_name, "r+")


    mean = np.array(file["/obs_mean"]).astype("float32")
    var = np.array(file["/obs_var"]).astype("float32")

    return mean,var

model_path = os.path.abspath(os.getcwd())
data_stat_dir = os.path.abspath(os.getcwd())

obs_mean,obs_var = get_data_statistics(data_stat_dir,"data_stat.h5")
navigator = Navigator(len(obs_mean),29,model_path,obs_mean,obs_var)
navigator.connect()

while True:
    navigator.step()

    if navigator.terminate:
        break
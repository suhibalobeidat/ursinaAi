
import imp
import pickle
import cv2
from cv2 import Param_INT
import numpy as np
import h5py
from PIL import Image
import glob
import os
import argparse
import torch
import torchvision.transforms as transforms
from pathlib import Path
import os.path
import yaml

def get_image(args,parallel_workers,workers_ids,done, init = True):

    return np.zeros(shape = (parallel_workers,1,1,1))
    
    if init:
        plan_inputs,plan_inputs_ids=load_images(glob.glob(args.initplanimage),workers_ids,args,args.initplanimage,done)
    else:
        plan_inputs,plan_inputs_ids=load_images(glob.glob(args.planimage),workers_ids,args,args.planimage,done)


    """ output = np.empty(shape = (parallel_workers,100,100,3))

    for i in range(len(workers_ids)):
        for j in range(len(plan_inputs_ids)):
            #print(f"worker id: {workers_ids[i]}, input id: {plan_inputs_ids[j]}")
            if workers_ids[i] == plan_inputs_ids[j]:
                output[i] = plan_inputs[j]
    plan_outputs = output.copy() """
    plan_outputs = plan_inputs
    return plan_outputs

def load_images(image_files,workers_id,args,args_option,done):
    ids = []
    #global i

    #num_images = 0
    if args_option == args.planimage:
        path = "C:/Users/sohai/Desktop/GAIL/planimages"
    elif args_option == args.initplanimage:
        path = "C:/Users/sohai/Desktop/GAIL/initplanimages"
    #num_images = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
    loaded_images = []
    image_name = None
    image_loaded = False
    worker_id = workers_id[0]
    while image_loaded == False:
        for file in image_files:
            id = get_image_id(file)
            if id != worker_id:
                continue
            image_name = file
            image = np.asarray(Image.open(file), dtype=float) 
            image = image.astype(np.uint8)

            if image.shape[0] != 84:
                difference = 84 - image.shape[0]
                for _ in range(difference):
                    row = image[-1,:,:]
                    row = row[np.newaxis,:,:]
                    image = np.append(image,row,axis= 0)

            if image.shape[1] != 84:
                difference = 84 - image.shape[1]
                for _ in range(difference):
                    row = image[:,-1,:]
                    row = row[:,np.newaxis,:]
                    image = np.append(image,row,axis= 1) 

            loaded_images.append(image)
            id = get_image_id(file)
            ids.append(id)
            image_loaded = True
    #clear_folder(args_option,args)
    clear_image(args_option,args,image_name)
    #i+=1
    #save_image(f'C:/Users/sohai/Desktop/GAIL/expertimages/{i}.png', image)
    #return (np.stack(loaded_images, axis=0),ids)
    return (loaded_images[0],ids)

def clear_image(args_option,args,image_name):

    #print(f"Delete {image_name}")

    if args_option == args.planimage:
        path = "C:/Users/sohai/Desktop/GAIL/planimages"
    elif args_option == args.initplanimage:
        path = "C:/Users/sohai/Desktop/GAIL/initplanimages"

    for i in glob.glob(args_option):
        if i != image_name:
            continue
        try:
            os.chmod(i,0o777)
            os.remove(i)
        except OSError:
            pass


i = 0
def process_data(image,label,text_input_length,action_direction_length,data_stat):
    #global i
    #i+=1
    #save_image(f'C:/Users/sohai/Desktop/GAIL/expertimages/{i}.png', image) 
    #print("labels shape", len(label.shape))
    #print("image_input shape", len(image.shape))
    if len(label.shape) == 1:
        labels = np.array([label])
        image_input = np.array([image])
    else:
        labels = np.array(label)
        image_input = np.array(image)



    images_mean = data_stat["images_mean"]
    images_std = data_stat["images_std"]
    texts_mean = data_stat["texts_mean"]
    texts_std = data_stat["texts_std"]

    #print("image mean", images_mean)
    #print("image std", images_std)


    #image_input = (image_input - images_mean) / images_std

    texts_input,actions_direction,actions_length,actions_mask = split_data(labels,text_input_length,action_direction_length)

    #print(f"text input shape{texts_input.shape}")
    #texts_input = (texts_input - texts_mean)/texts_std
    actions_length = actions_length/10

    actions_length = np.clip(actions_length,0,0.99999)

    texts_input = torch.FloatTensor(texts_input)
    actions_direction = torch.FloatTensor(actions_direction)
    actions_length = torch.FloatTensor(actions_length)
    image_input = torch.FloatTensor(image_input)#.permute(0,3,1,2)
    actions_mask = torch.BoolTensor(actions_mask)

    #print(f"images shape {image_input.shape}")
    #print(f"texts shape {texts_input.shape}")
    
    return image_input.squeeze(),texts_input.squeeze(),actions_direction.squeeze(),actions_length.squeeze(),actions_mask.squeeze()

def split_data(texts,input_text_len,action_direction_len):
    text_input = texts[:,0:input_text_len]#145 input
    action_mask = texts[:,input_text_len:input_text_len + action_direction_len]#25 action direction
    action_direction = texts[:,texts.shape[1]-26:texts.shape[1]-1] #25 action mask
    action_length = texts[:,[texts.shape[1]-1]] # action legnth

    #print(text_input.shape)
    #print(action_mask.shape)
    #print(action_direction.shape)
    #print(action_length.shape)
    return text_input,action_direction,action_length,action_mask

def get_data_statistics(file_name):
    file = h5py.File(file_name, "r+")


    texts_mean = np.array(file["/texts_mean"]).astype("float32")
    texts_std = np.array(file["/texts_std"]).astype("float32")


    print(f"texts_mean {texts_mean.shape}")
    print(f"texts_std {texts_std.shape}")


    return 127.5,127.5,texts_mean,texts_std
    
def get_data_statistics_gym(file_name):
    file = h5py.File(file_name, "r+")

    texts_mean = np.array(file["/texts_mean"]).astype("float32")
    texts_std = np.array(file["/texts_std"]).astype("float32")

    print(f"texts_mean {texts_mean.shape}")
    print(f"texts_std {texts_std.shape}")


    return None,None,texts_mean,texts_std
def convert_while_pixels_to_black(img):
    white_pixels = np.where(#image shape  (height, width, channels)
    (img[:,:, :, 0] == 255) & 
    (img[:,:, :, 1] == 255) & 
    (img[:,:, :, 2] == 255)
    )

    # set those pixels to white
    img[white_pixels] = [0, 0, 0]

    return img

def get_image_id(image_name):
    list = image_name
    list = list.split("\\")
    list = list[1].split("-")
    id = int(list[0].strip())
    return id
def check_done_reward(done,reward,total_sucss):
    done_bool = np.empty((done.shape[0],1))
    for i in range(len(done)):
        #done_bool[i] = 0 if reward[i] > 0 and reward[i] < 1 else float(done[i])
        if done[i] == 2:
            done_bool[i] = 0 
            done[i] = 1
        elif done[i] == 3:
            done_bool[i] = 1
            done[i] = 1
        else:
            done_bool[i] = float(done[i])

        if reward[i] == 100:
            if total_sucss:
                total_sucss[i] +=1
    return done_bool

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

def correct_done_drqv2(done):
    done_bool = 0
    successful_ep = False
    new_done = done
    if done == 2:
        done_bool = 0 
        new_done = 1
        successful_ep = False
    elif done == 3:
        done_bool = 1
        new_done = 1
        successful_ep = True
    else:
        done_bool = done
        new_done = done
        successful_ep = False

    return new_done,done_bool,successful_ep
def merge_images(images, next_images,done):
    new_images = images
    for i in range(new_images.shape[0]):
        if done[i]:
            new_images[i] = next_images[i]   
    return new_images 


def action_directions_one_hot_vector(action_direction):
    """ directions = 25*[0]
    print(action_direction[0].item())
    directions[action_direction[0].item()] = 1
    return directions """

    t_like = torch.randint(0,1,size = (action_direction.shape[0],25))

    for i in range(t_like.shape[0]):
        t_like[i][action_direction[i].item()] = 1
    return t_like


def normalize_data(image_input,text_input, data_stat):
    image_input = normalize(image_input,data_stat["images_mean"],data_stat["images_std"])
    text_input = normalize(text_input,data_stat["texts_mean"],data_stat["texts_std"])

    return image_input,text_input


def normalize(input, mean, std):
    """ if input[0].shape == (100,100,3):
        input = convert_while_pixels_to_black(input) """

    input = np.clip((input - mean)/std,-10,10)
    return input

def convert_to_gray_scale(images):
    images = np.mean(images,axis=3)
    images = images/255
    images = (images - 0.5)/0.5
    return images

def images_args():
    import argparse
    parser = argparse.ArgumentParser(description='state images')
    parser.add_argument('--planimage', default='C:/Users/sohai/Desktop/GAIL/planimages/*.png', type=str, help='Input filename or folder.')
    parser.add_argument('--initplanimage', default='C:/Users/sohai/Desktop/GAIL/initplanimages/*.png', type=str, help='Input filename or folder.')
    parser.add_argument('--testImages', default='C:/Users/sohai/Desktop/GAIL/img/*.png', type=str, help='Input filename or folder.')

    args = parser.parse_args()
    return args

def save_image(filename, outputs):
    im = Image.fromarray(np.uint8(outputs))
    im.save(filename)

def save_model(model ,filename, directory):
    torch.save(model, '%s/%s.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
def load_model(_model,filename, directory):
    _model.load_state_dict(torch.load('%s/%s.pth' % (directory, filename)))

def clear_folder(args_option,args):

    if args_option == args.planimage:
        path = "C:/Users/sohai/Desktop/GAIL/planimages"
    elif args_option == args.initplanimage:
        path = "C:/Users/sohai/Desktop/GAIL/initplanimages"
    elif args_option == args.testImages:
        path = "C:/Users/sohai/Desktop/GAIL/img"

    num_files = 100
    while num_files != 0:
        num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
        for i in glob.glob(args_option):
            try:
                os.chmod(i,0o777)
                os.remove(i)
            except OSError:
                pass

def create_data_stat(texts_mean,texts_std):
    file = h5py.File("data_stat.h5", "w")

    file.create_dataset(
            "texts_mean", np.shape(texts_mean), data=texts_mean
        )
    file.create_dataset(
            "texts_std", np.shape(texts_std), data=texts_std
        )
    file.close()


import random
import numpy as np
import math
from math import cos,sin
from ursina import Vec3,Entity,Mesh

from enums import Directions, RotationAxis

def vec_len(arr):
    return np.sqrt( arr[0]**2 + arr[1]**2 + arr[2]**2 )

def get_distance(vec1,vec2):
    return np.sqrt( (vec1.x - vec2.x)**2 + (vec1.y - vec2.y)**2 + (vec1.z - vec2.z)**2 )


def vec3_len(vec):
    return np.sqrt( vec.x**2 + vec.y**2 + vec.z**2 )

def project_vec(vec, direction):
    new_vec = direction * np.dot(vec, direction) / np.dot(direction, direction)
    return np_to_vec3(new_vec)


def deg_to_rad(angle):
    return (angle*math.pi/180)

def rad_to_deg(angle):
    return (angle *180/math.pi)

def np_to_vec3(vec):
    if len(vec.shape) == 1:
        new_vec = Vec3(vec[0], vec[1], vec[2])
    elif len(vec.shape) == 2:
        new_vec = Vec3(vec[0,0], vec[0,1], vec[0,2])

    return new_vec

def vec3_to_np(vec):
    return np.array([vec.x, vec.y, vec.z])

def vec3_to_list(vec):
    return [vec.x, vec.y, vec.z]

def rotate_vec(angle, rotation_axis,vec):
    rad = deg_to_rad(angle)

    if rotation_axis == RotationAxis.Z:
        rotation_matrix = [[cos(rad),-sin(rad),0],[sin(rad),cos(rad),0],[0,0,1]]
    elif rotation_axis == RotationAxis.Y:
        rotation_matrix = [[cos(rad),0,sin(rad)],[0,1,0],[-sin(rad),0,cos(rad)]]
    elif(rotation_axis == RotationAxis.X):
        rotation_matrix = [[1,0,0],[0,cos(rad),-sin(rad)],[0,sin(rad),cos(rad)]]

    rotation_matrix = rotation_matrix

    vec = [vec.x,vec.y,vec.z]

    new_vec = np.matmul(rotation_matrix,vec)

    new_vec = np_to_vec3(new_vec)

    return new_vec

def rotate_Vec_around_vec(angle, base_vec, vec):
    rad = deg_to_rad(angle)
    vec = vec3_to_np(vec)
    base_vec = vec3_to_np(base_vec)

    new_vec = ((1-np.cos(rad)) * np.dot(vec,base_vec)*base_vec) \
        + (np.cos(rad)*vec + np.sin(rad)*(np.cross(vec,base_vec)))

    new_vec = np_to_vec3(new_vec)
    
    return new_vec

def cross_product(vec1,vec2):
    vec1 = vec3_to_np(vec1)
    vec2 = vec3_to_np(vec2)

    perp_vec = np.cross(vec1,vec2)

    perp_vec = np_to_vec3(perp_vec)
    return perp_vec


def angle_between_vec(vec1, vec2):

    dot = np.dot(vec1,vec2)

    vec1_length = vec_len(vec1)
    vec2_length = vec_len(vec2)

    angle = dot/(vec1_length*vec2_length)
    angle = np.arccos(angle)

    return angle

def get_elbow_length(angle, width, radius_mul):
    angle = deg_to_rad(angle)
    return (radius_mul*width)*np.tan(angle/2)
    
def are_vectors_equal(vec1,vec2):
    if vec1.x == vec2.x and vec1.y == vec2.y and vec1.z == vec2.z:
        return True
    
    return False

def correct_value(x,x0,x1,y0,y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def get_random_number(minValue, maxValue):
    return random.random() * (maxValue - minValue) + minValue

def get_random_vec3():
    x = random.randint(0,50) * random.random()
    y = random.randint(0,50) * random.random()
    z = random.randint(0,50) * random.random()

    return Vec3(x,y,z)

def get_point_at_distance(distance):
    vec = get_random_vec3()
    vec = vec/vec3_len(vec) 
    point = vec * distance

    return point

def from_mm_to_ft(value):
    new_value = (value/304.8)
    return new_value

def from_ft_to_mm(value):
    new_value = (value*304.8)
    return new_value

def direction_vec(direction):
    if direction == Directions.Positive_x:
        return Vec3(1,0,0)
    elif direction == Directions.Negative_x:
        return Vec3(-1,0,0)
    elif direction == Directions.Positive_y:
        return Vec3(0,1,0)
    elif direction == Directions.Negative_y:
        return Vec3(0,-1,0)
    elif direction == Directions.Positive_z:
        return Vec3(0,0,1)
    elif direction == Directions.Negative_z:
        return Vec3(0,0,-1)

def get_direction_number(vec):
    if vec.x == 1:
        return Directions.Positive_x
    elif vec.x == -1:
        return Directions.Negative_x
    elif vec.y == 1:
        return Directions.Positive_y
    elif vec.y == -1:
        return Directions.Negative_y
    elif vec.z == 1:
        return Directions.Positive_z
    elif vec.z == -1:
        return Directions.Negative_z 

def get_dim_at_dir(x_dim,y_dim,z_dim,direction):
    if direction == Directions.Positive_x:
        return x_dim
    elif direction == Directions.Negative_x:
        return x_dim
    elif direction == Directions.Positive_y:
        return y_dim
    elif direction == Directions.Negative_y:
        return y_dim
    elif direction == Directions.Positive_z:
        return z_dim
    elif direction == Directions.Negative_z:
        return z_dim

def draw_line(point1, point2):
    points = [point1,point2]
    line = Entity(model=Mesh(vertices=points, mode='line'))
    return line
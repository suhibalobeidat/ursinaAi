
from tkinter import Y
from ursina import *
from enums import MovmentDirection, RotationAxis
from utils_ursina import *
import numpy as np
from transform import Transform
from panda3d.core import LMatrix4f as matrix

class Segmant(Entity):
    def __init__(self, shape, start_position, end_position, width, height,parent_segmant, segmant_count,angle,relative_movement_direction,x_axis,y_axis,z_axis,length ,radius_mul = 1.5):
        super().__init__(
            model = "cube",
            color = color.red,
            texture  = "white_cube",
            collision  = True
        )

        self.parent_semant = parent_segmant
        self.segmant_count = segmant_count
        self.angle = angle
        self.start_position = start_position
        self.end_position = end_position
        self.width = width
        self.height = height
        self.max_dim = max(self.width,self.height)
        self.max_elbow_length = get_elbow_length(90,self.max_dim,radius_mul)
        self.shape = shape
        self.radius_mul = radius_mul
        self.relative_movement_direction = relative_movement_direction#relative movment direction to parent segmant
        if relative_movement_direction == MovmentDirection.X:
            self.elbow_length = get_elbow_length(abs(self.angle),self.width,radius_mul)
        elif relative_movement_direction.Y:
            self.elbow_length = get_elbow_length(abs(self.angle),self.height,radius_mul)
        else:
            self.elbow_length = 0
        self.length = length
        self.net_length = self.length - self.elbow_length
        self.position = self.start_position + z_axis*(self.length/2)

        if x_axis:
            m = matrix()

            m.setRow(0,x_axis * self.width)
            m.setRow(1,y_axis * self.height)
            m.setRow(2,z_axis * self.length)
            m.setRow(3,self.position)

            self.setMat(m)

        else: 
            self.scale_x = width
            self.scale_y = height
            self.scale_z = self.length
            


        

    def set_axis(self, x_axis, y_axis, z_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.movment_transform = Transform(self.x_axis,self.y_axis,self.z_axis,self.end_position)

    

def create_start_segmant(shape, width, height,length,radius_mul = 1.5):
    start_segmant = Segmant(shape = shape, start_position = Vec3(0,0,0),end_position= Vec3(0,0,1)*length,width = width,height = height,parent_segmant=None,segmant_count=1,angle = 0,relative_movement_direction = MovmentDirection.Z,x_axis=None,y_axis=None,z_axis=Vec3(0,0,1),length=length,radius_mul=radius_mul)
    body_transform = start_segmant.getMat()
    x_axis = body_transform.getRow3(0).normalized()
    y_axis = body_transform.getRow3(1).normalized()
    z_axis = body_transform.getRow3(2).normalized()

    x_axis = Vec3(x_axis.x,x_axis.y,x_axis.z)
    y_axis = Vec3(y_axis.x,y_axis.y,y_axis.z)
    z_axis = Vec3(z_axis.x,z_axis.y,z_axis.z)

    start_segmant.set_axis(x_axis,y_axis,z_axis)

    start_segmant.collider = "box"

    return start_segmant

def create_segmant(parent_segmant,end_position,angle, direction, length,radius_mul = 1.5):
    
    if direction == MovmentDirection.X:
        x_axis = rotate_Vec_around_vec(angle,parent_segmant.y_axis,parent_segmant.x_axis)
        y_axis = parent_segmant.y_axis
        z_axis = rotate_Vec_around_vec(angle,parent_segmant.y_axis,parent_segmant.z_axis)
    elif direction == MovmentDirection.Y:
        x_axis = parent_segmant.x_axis
        y_axis = rotate_Vec_around_vec(angle,parent_segmant.x_axis,parent_segmant.y_axis)
        z_axis = rotate_Vec_around_vec(angle,parent_segmant.x_axis,parent_segmant.z_axis)
    elif direction == MovmentDirection.Z:
        x_axis = parent_segmant.x_axis
        y_axis = parent_segmant.y_axis
        z_axis = parent_segmant.z_axis


    new_segmant = Segmant(parent_segmant.shape,deepcopy(parent_segmant.end_position),
    end_position,parent_segmant.width,parent_segmant.height,parent_segmant=parent_segmant,
    segmant_count=parent_segmant.segmant_count+1,angle=angle,relative_movement_direction=direction,
    x_axis = x_axis,y_axis = y_axis,z_axis = z_axis, length=length,radius_mul=radius_mul)

    new_segmant.set_axis(x_axis,y_axis,z_axis)

    new_segmant.collider = "box"


    return new_segmant

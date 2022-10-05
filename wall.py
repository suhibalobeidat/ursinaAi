from ursina import *
from panda3d.core import LMatrix4f as matrix


class Wall(Entity):
    def __init__(self, width, height,transform,thickness =0.1, color = color.white,parent_wall_image = None):
        super().__init__(
            model = "cube",
            color = color,
            texture  = "white_cube",
            collision  = True
        )

        self.width = width
        self.height = height
        self.thickness = thickness
        self.wall_transform = transform
        self.wall_origin = transform.origin
        self.parent_wall_image = parent_wall_image
        self.position = transform.origin

        m = matrix()

        m.setRow(0,transform.x_base * self.width)
        m.setRow(1,transform.y_base * self.height)
        m.setRow(2,transform.z_base * self.thickness)
        m.setRow(3,self.position)

        self.setMat(m)

        self.sub_walls = []

        self.collider = "box"


    def update_origin(self,new_origin):
        self.wall_origin = new_origin
        self.position = new_origin
        self.wall_transform.origin = new_origin

class WallImage():
        def __init__(self,origin, matrix,transform,width,height,thickness):

            self.width = width
            self.height = height
            self.thickness = thickness
            self.wall_transform = transform
            self.origin = origin
            self.matrix = matrix


def create_wall_image(wall):
    return WallImage(wall.origin,wall.getMat(),wall.wall_transform,wall.width,wall.height,wall.thickness)

def create_wall_from_rect(rect, color):
    return Wall(rect.width,rect.height,rect.transform, color=color)

def create_wall_from_image(wall_image):
    return Wall(wall_image.width,wall_image.height,wall_image.wall_transform,wall_image.thickness)
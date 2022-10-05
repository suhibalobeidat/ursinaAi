from enums import *
from ursina import Vec3

class Rect:
    def __init__(self,origin, min_point, max_point,width,height, transform):
        self.origin = origin
        self.min = min_point
        self.max = max_point
        self.width = width
        self.height = height
        self.area = self.width * self.height     
        self.transform = transform
        self.const_dim = self.get_const_dim()

    def get_const_dim(self):
        if self.min.x == self.max.x:
            return Base.X
        elif self.min.y == self.max.y:
            return Base.Y
        elif self.min.z == self.max.z:
            return Base.Z
        else:
            return None
    
        
def create_rect(opening):
    origin = opening.origin
    width = opening.width
    height = opening.height
    max = origin +  (opening.opening_transform.x_base * width/2) +  (opening.opening_transform.y_base * height/2)
    #point2 = origin +  (opening.opening_transform.x_base * width/2 -  opening.opening_transform.y_base * height/2)
    #point3 = origin +  (-opening.opening_transform.x_base * width/2 +  opening.opening_transform.y_base * height/2)
    min = origin +  (-opening.opening_transform.x_base * width/2) -  (opening.opening_transform.y_base * height/2)

    return Rect(origin,min,max,width,height,opening.opening_transform)

def get_overlap_rect(rect1,rect2):
    #print(f"rect1 const dim {rect1.const_dim}, rect2 const dim {rect2.const_dim}")
    if rect1.const_dim == rect2.const_dim == Base.X:
        max_y = min(rect1.max.y,rect2.max.y)
        max_z = min(rect1.max.z,rect2.max.z)
        max_point = Vec3(rect1.max.x, max_y,max_z)

        min_y = max(rect1.min.y,rect2.min.y)
        min_z = max(rect1.min.z,rect2.min.z)
        min_point = Vec3(rect1.min.x, min_y,min_z)

        origin = (max_point + min_point) /2

        width = min(rect1.max.y, rect2.max.y) - max(rect1.min.y, rect2.min.y)
        height = min(rect1.max.z, rect2.max.z) - max(rect1.min.z, rect2.min.z)

        return Rect(origin,min_point,max_point,width,height,rect1.transform)

    elif rect1.const_dim == rect2.const_dim == Base.Y:
        max_x = min(rect1.max.x,rect2.max.x)
        max_z = min(rect1.max.z,rect2.max.z)
        max_point = Vec3(max_x,rect1.max.y,max_z)

        min_x = max(rect1.min.x,rect2.min.x)
        min_z = max(rect1.min.z,rect2.min.z)
        min_point = Vec3(min_x,rect1.min.y,max_z)

        origin = (max_point + min_point) /2

        width = min(rect1.max.x, rect2.max.x) - max(rect1.min.x, rect2.min.x)
        height = min(rect1.max.z, rect2.max.z) - max(rect1.min.z, rect2.min.z)

        return Rect(origin,min_point,max_point,width,height,rect1.transform)

    elif rect1.const_dim == rect2.const_dim == Base.Z:
        max_x = min(rect1.max.x,rect2.max.x)
        max_y = min(rect1.max.y,rect2.max.y)
        max_point = Vec3(max_x,max_y,rect1.max.z)

        min_x = max(rect1.min.x,rect2.min.x)
        min_y = max(rect1.min.y,rect2.min.y)
        min_point = Vec3(min_x,min_y,rect1.min.z)

        origin = (max_point + min_point) /2

        width = min(rect1.max.x, rect2.max.x) - max(rect1.min.x, rect2.min.x)
        height = min(rect1.max.y, rect2.max.y) - max(rect1.min.y, rect2.min.y)

        return Rect(origin,min_point,max_point,width,height,rect1.transform)
from wall import Wall
from ursina import color,Vec3

class Transform():
    def __init__(self, xBase = None, yBase = None, zBase = None,origin = None):

        if xBase == None:
            xBase = Vec3(1,0,0)
        
        if yBase == None:
            yBase = Vec3(0,1,0)

        if zBase == None:
            zBase = Vec3(0,0,1)

        if origin == None:
            origin = Vec3(0,0,0)

        self.x_base = xBase
        self.y_base = yBase
        self.z_base = zBase
        self.origin = origin

        self.x_base_l = [self.x_base.x,self.x_base.y,self.x_base.z]
        self.y_base_l = [self.y_base.x,self.y_base.y,self.y_base.z]
        self.z_base_l = [self.z_base.x,self.z_base.y,self.z_base.z]

        self.x_line = None
        self.y_line = None
        self.z_line = None

    def draw(self, x_base = True, y_base = True, z_base = True,thikness = 1, length = 1):
        if x_base:
            t = Transform(self.y_base,self.z_base,self.x_base,self.origin)
            self.x_line = Wall(thikness,thikness,t,length,color.blue)
            self.x_line.position += self.x_line.wall_transform.z_base*(length/2)    

        if y_base:
            t = Transform(self.x_base,self.z_base,self.y_base,self.origin)
            self.y_line = Wall(thikness,thikness,t,length,color.yellow)
            self.y_line.position += self.y_line.wall_transform.z_base*(length/2)   

        if z_base:
            t = Transform(self.x_base,self.y_base,self.z_base,self.origin)   
            self.z_line = Wall(thikness,thikness,t,length,color.green)
            self.z_line.position += self.z_line.wall_transform.z_base*(length/2)   


    

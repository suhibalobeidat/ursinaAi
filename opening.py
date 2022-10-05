from transform import Transform


from transform import Transform
class Opening():
    def __init__(self, xBase, yBase, zBase,origin,width,height, parent_wall_image = None):
        self.x_base = xBase
        self.y_base = yBase
        self.z_base = zBase
        self.origin = origin


        self.opening_transform = Transform(self.x_base,self.y_base,self.z_base,self.origin)

        self.width = width
        self.height = height

        self.parent_wall_image = parent_wall_image

    def update_origin(self,new_origin):
        self.origin = new_origin
        self.opening_transform.origin = new_origin
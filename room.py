from operator import truediv
from opening import Opening
from wall import *
from transform import Transform
from enums import *
from ursina import *
from utils_ursina import *
from rectangle import *
import random
from utils_ursina import get_dim_at_dir
class Room():
    def __init__(self,origin,x_dim,y_dim,height, number):
        self.origin = origin
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.height = height
        self.walls = []
        self.valid_walls = []
        self.non_opening_walls = []
        self.opening_walls = []
        self.walls_images = []
        self.openings = []
        self.common_openings = []
        self.number = number
        self.faces_number = 6
        self.create_walls()

    def is_inside(self,point):
        if point.x >= (self.origin - (Vec3(1,0,0) * 0.5 * self.x_dim)).x and point.x <= (self.origin + (Vec3(1,0,0) * 0.5 * self.x_dim)).x:
            if point.y >= (self.origin - (Vec3(0,1,0) * 0.5 * self.y_dim)).y and point.y <= (self.origin + (Vec3(0,1,0) * 0.5 * self.y_dim)).y:
                if point.z >= (self.origin - (Vec3(0,0,1) * 0.5 * self.height)).z and point.z <= (self.origin + (Vec3(0,0,1) * 0.5 * self.height)).z:
                    return True
        return False    

    def sample_point(self):
        random_x = random.random() - 0.5
        random_y = random.random() - 0.5
        random_z = random.random() - 0.5

        sample_point = self.origin + (Vec3(1,0,0) * random_x * self.x_dim)
        sample_point = sample_point + (Vec3(0,1,0) * random_y  * self.y_dim)
        sample_point = sample_point + (Vec3(0,0,1) * random_z  * self.height)

        return sample_point

    def create_walls(self):
        for i in range(self.faces_number):  
            width,height = self.get_wall_dim(Directions(i))
            wall_transform = self.get_wall_transform(Directions(i))
            wall = Wall(width,height,wall_transform)
            self.walls.append(wall)
            self.non_opening_walls.append(wall)
            self.valid_walls.append(wall)

    def get_relative_to_wall(self,wall,opening):
        if opening.origin == None:
            if self.number == 1:
                return (0.5,0.5)
            relative_width = random.random()
            relative_height = random.random()

            return (relative_width,relative_height)
        else:
            wall_left = wall.wall_origin - (wall.wall_transform.x_base * wall.width/2) - (wall.wall_transform.y_base * wall.height/2)
            opening_left = opening.origin +  project_vec(vec3_to_np(wall_left - opening.origin),vec3_to_np(wall.wall_transform.y_base))
            opening_left -= (wall.wall_transform.x_base * opening.width/2)
            left_width = get_distance(opening_left,wall_left)       
            remaining_wall_width = wall.width - opening.width
            relative_width = left_width/remaining_wall_width

            wall_up = wall.wall_origin - (wall.wall_transform.x_base * wall.width/2) - (wall.wall_transform.y_base * wall.height/2)
            opening_up = opening.origin +  project_vec(vec3_to_np(wall_up - opening.origin),vec3_to_np(wall.wall_transform.x_base))
            opening_up -= (wall.wall_transform.y_base * opening.height/2)
            upper_height = get_distance(opening_up,wall_up)
            remaining_wall_height = wall.height - opening.height
            relative_height = upper_height/remaining_wall_height
            relative_height = 1- relative_height

            #print(f"relative width {relative_width}, relative height {relative_height}")
            #return (0.5,0.5)
            return (relative_width,relative_height)      

    def get_wall_transform(self,direction):

        if direction == Directions.Positive_x:
            return Transform(Vec3(0,1,0),Vec3(0,0,1),Vec3(1,0,0),self.origin + (Vec3(1,0,0)*self.x_dim/2))
        elif direction == Directions.Negative_x:
            return Transform(Vec3(0,1,0),Vec3(0,0,1),Vec3(-1,0,0),self.origin + (Vec3(-1,0,0)*self.x_dim/2))
        elif direction == Directions.Positive_y:
            return Transform(Vec3(1,0,0),Vec3(0,0,1),Vec3(0,1,0),self.origin + (Vec3(0,1,0)*self.y_dim/2))
        elif direction == Directions.Negative_y:
            return Transform(Vec3(1,0,0),Vec3(0,0,1),Vec3(0,-1,0),self.origin + (Vec3(0,-1,0)*self.y_dim/2))
        elif direction == Directions.Positive_z:
            return Transform(Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1),self.origin + (Vec3(0,0,1)*self.height/2))
        elif direction == Directions.Negative_z:
            return Transform(Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,-1),self.origin + (Vec3(0,0,-1)*self.height/2))

    def get_wall_dim(self,direction):
        if direction == Directions.Positive_x:
            return(self.y_dim,self.height)
        elif direction == Directions.Negative_x:
            return(self.y_dim,self.height)
        elif direction == Directions.Positive_y:
            return(self.x_dim,self.height)
        elif direction == Directions.Negative_y:
            return(self.x_dim,self.height)
        elif direction == Directions.Positive_z:
            return(self.x_dim,self.y_dim)
        elif direction == Directions.Negative_z:
            return(self.x_dim,self.y_dim)

    def move_room(self,direction,distance = None):
        if distance:
            for wall in self.walls:
                new_origin = wall.position
                new_origin += (direction*distance)
                wall.update_origin(new_origin) 
    

            for opening in self.openings:
                new_origin = opening.origin
                new_origin += (direction*distance)
                opening.update_origin(new_origin)  

            self.origin += (direction*distance)
        else:
            for wall in self.walls:
                new_origin = wall.position
                new_origin += direction
                wall.update_origin(new_origin) 


            for opening in self.openings:
                new_origin = opening.origin
                new_origin += direction
                opening.update_origin(new_origin)  

            self.origin += direction     


    def correct_wall(self,opening,wall):
        relative_width,relative_height = self.get_relative_to_wall(wall,opening)


        wall_width = wall.width
        wall_height = wall.height
        wall_transform = wall.wall_transform
        wall_origin = wall_transform.origin

        remaining_wall_width = wall_width - opening.width
        left_width = remaining_wall_width * relative_width
        right_width = remaining_wall_width - left_width


        remaining_wall_height= wall_height - opening.height
        upper_height = remaining_wall_height * relative_height
        lower_height = remaining_wall_height - upper_height

        wall_image = None#create_wall_image(wall)

        t = Transform(wall_transform.x_base,wall_transform.y_base,wall_transform.z_base,wall_origin - (wall_transform.x_base*(wall_width/2 - left_width/2)))
        self.walls.append(Wall(left_width,wall_height,t,parent_wall_image=wall_image))


        t = Transform(wall_transform.x_base,wall_transform.y_base,wall_transform.z_base,wall_origin + (wall_transform.x_base*(wall_width/2 - right_width/2)))
        self.walls.append(Wall(right_width,wall_height,t,parent_wall_image=wall_image))

        t = Transform(wall_transform.x_base,wall_transform.y_base,wall_transform.z_base,wall_origin + (wall_transform.y_base*(wall_height/2 - upper_height/2)))
        self.walls.append(Wall(wall_width,upper_height,t,parent_wall_image=wall_image))

        t = Transform(wall_transform.x_base,wall_transform.y_base,wall_transform.z_base,wall_origin - (wall_transform.y_base*(wall_height/2 - lower_height/2)))
        self.walls.append(Wall(wall_width,lower_height,t,parent_wall_image=wall_image)) 

        destroy(wall)
        self.walls.remove(wall)
        self.non_opening_walls.remove(wall) 

    def create_common_opening(self,opening1, opening2):
        rect1 = create_rect(opening1)
        rect2 = create_rect(opening2)

        #wall = create_wall_from_rect(rect1,color.blue)
        #wall = create_wall_from_rect(rect2,color.yellow)
        try:
            overlap_rect = get_overlap_rect(rect1,rect2)

            wall = create_wall_from_rect(overlap_rect,color.red)
        except:
            pass

        common_opening = Opening(wall.wall_transform.x_base,wall.wall_transform.y_base,wall.wall_transform.z_base
                                ,wall.position,wall.width,wall.height)
        
        self.common_openings.append(common_opening)

        destroy(wall)
        return common_opening

        


    def create_wall_opening(self,wall,opening):
            #print("opening origin", opening.origin)
            relative_width,relative_height = self.get_relative_to_wall(wall,opening)
            self.create_opening(wall,opening.width,opening.height,relative_width,relative_height,opening)
            self.opening_walls.append(wall)
            self.valid_walls.remove(wall)
            self.openings.append(opening)

    def create_opening(self,wall,width,height,percent_net_width,percent_net_height,opening):
        wall_width = wall.width
        wall_height = wall.height
        wall_transform = wall.wall_transform
        wall_origin = wall_transform.origin

        remaining_wall_width = wall_width - width
        left_width = remaining_wall_width * percent_net_width
        right_width = remaining_wall_width - left_width


        remaining_wall_height= wall_height - height
        upper_height = remaining_wall_height * percent_net_height
        lower_height = remaining_wall_height - upper_height

        wall_image = create_wall_image(wall)

        self.walls_images.append(wall_image)

        opening.parent_wall_image = wall_image

        if opening.origin == None:
            origin = wall_origin - (wall_transform.x_base*(wall_width/2 - left_width - width/2))
            origin = origin - (wall_transform.y_base*(wall_height/2 - lower_height - height/2))
            opening.update_origin(origin)

    def get_wall_at_direction(self,direction, angle_offset):
        for wall in self.valid_walls:
            angle  = angle_between_vec(wall.wall_transform.z_base, direction)  
            #print(rad_to_deg(angle))
            if rad_to_deg(angle) == angle_offset:
                return wall


    def delete(self):
        for wall in self.walls:
            destroy(wall)
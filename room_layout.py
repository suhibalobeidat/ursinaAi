from enums import *
from ursina import *
from opening import Opening
from rectangle import create_rect
from transform import Transform
from utils_ursina import *
from room import Room
import random
class Layout():
    def __init__(self,max_rooms_number = 5,min_distance = 5):
        self.mep_curve_segmant = None
        self.rooms = []
        self.max_room_dim = 25
        self.max_rooms_number = max_rooms_number
        self.min_distance = min_distance
        self.goal_room = None
        self.path_rects = []
        self.next_rect = None
        self.current_room = None
        self.is_last_room = False

    def get_path_rects(self):

        for i in range(self.max_rooms_number-1):         
            common_openings = self.rooms[i+1].common_openings[-1]
            rect = create_rect(common_openings)
            self.path_rects.append(rect)

        self.next_rect = self.path_rects[0]
        self.previous_rect = self.path_rects[0]
        self.current_room = self.rooms[0]

    def get_next_rect(self,segmant):
        if self.is_inside_last_room(segmant):
            self.is_last_room = True
            self.previous_rect = self.next_rect
            return self.previous_rect,self.path_rects[-1], True
            
        
        if self.rooms[self.current_room.number].is_inside(segmant.movment_transform.origin):
            self.previous_rect = self.next_rect
            self.next_rect = self.path_rects[self.current_room.number] 
            self.current_room = self.rooms[self.current_room.number]

            return self.previous_rect,self.next_rect, True
        else:
            return self.previous_rect,self.next_rect,False

            



    def is_inside_last_room(self,segmant):
        if self.rooms[self.max_rooms_number-2].is_inside(segmant.movment_transform.origin):
            return True
    
        return False
 


    def get_goal_point(self,distance):
        total_distance = 0

        goal_room = None
        for room in self.rooms:
            #t = Transform(origin=room.origin)
            #t.draw(thikness=2,length=6)

            if room.number == 1:
                total_distance += get_distance(room.origin,room.openings[-1].origin)
            elif len(room.openings) == 1:
                total_distance += 2*get_distance(room.openings[0].origin,room.origin)
            else:
                total_distance += get_distance(room.openings[0].origin,room.openings[1].origin)

            if total_distance >= distance:
                goal_room = room
                break
        
        if goal_room == None:
            goal_room = self.rooms[-1]

        while True:
            goal_point = goal_room.sample_point()
            if get_distance(goal_point,self.mep_curve_segmant.position) > self.min_distance:
                break

        self.goal_room = goal_room

        """ r = Transform(origin=goal_point)
        r.draw(thikness=2,length=6) """
        return goal_point



    def generate_layout(self, mep_curve_segmant,opening_size):
        self.mep_curve_segmant = mep_curve_segmant
        self.opening_size = opening_size

        #print(f"segmant width {self.mep_curve_segmant.width} , segmant height {self.mep_curve_segmant.height}, max_elbow_length {self.mep_curve_segmant.max_elbow_length}")
        for i in range(self.max_rooms_number):
            while True:
                x_dim, y_dim,height,needed_movment_direction = self.get_room_size()
                while True:
                    if len(self.rooms) > 1:
                        if len(self.rooms[-1].valid_walls) == 0:
                            break
                    origin,needed_movment_direction = self.get_room_origin(x_dim,y_dim,height,needed_movment_direction)
                    #print(f"x_dim {x_dim} , y_dim {y_dim} , height {height}")
                    is_succ = self.generate_room(origin,x_dim,y_dim,height,needed_movment_direction,i+1)

                    if is_succ:
                        break
                if is_succ:
                    break
                valid_walls = []
                valid_walls.extend(self.rooms[-1].non_opening_walls)
                self.rooms[-1].valid_walls = valid_walls
                #print("************************************************* NEW DIM")

        self.crerate_colliders()


    def get_room_origin(self,x_dim,y_dim,height, movment_direction = None):
        #print("number of rooms:", len(self.rooms))
        if len(self.rooms) == 0:
            #print("ROOM 1 ORIGIN: ", self.mep_curve_segmant.position)

            return self.mep_curve_segmant.position, movment_direction
        elif len(self.rooms) == 1:
            origin = self.rooms[-1].origin + self.rooms[-1].openings[-1].z_base* (get_dim_at_dir(self.rooms[-1].x_dim,self.rooms[-1].y_dim,self.rooms[-1].height,get_direction_number(self.rooms[-1].openings[-1].z_base))/2 + get_dim_at_dir(x_dim,y_dim,height,get_direction_number(self.rooms[-1].openings[-1].z_base))/2 )
            return origin, None

        else:
            #print("VALID WALLS", len(self.rooms[-1].valid_walls))
            movment_direction = self.rooms[-1].valid_walls[random.randint(0,len(self.rooms[-1].valid_walls)-1)].wall_transform.z_base
            origin = self.rooms[-1].origin + movment_direction* (get_dim_at_dir(self.rooms[-1].x_dim,self.rooms[-1].y_dim,self.rooms[-1].height,get_direction_number(movment_direction))/2 + get_dim_at_dir(x_dim,y_dim,height,get_direction_number(movment_direction))/2 )

            #origin = self.rooms[-1].origin + self.rooms[-1].openings[-1].z_base* (get_dim_at_dir(self.rooms[-1].x_dim,self.rooms[-1].y_dim,self.rooms[-1].height,get_direction_number(self.rooms[-1].openings[-1].z_base))/2 + get_dim_at_dir(x_dim,y_dim,height,get_direction_number(self.rooms[-1].openings[-1].z_base))/2 )

            #print("ROOM 2 ORIGIN: ", origin)

            return origin, get_direction_number(movment_direction)

    def get_room_opening(self,room,needed_movment_direction = None):
        if room.number == 1:
            rand_wall = room.get_wall_at_direction(direction_vec(needed_movment_direction),0)#room.walls[random.randint(0,len(room.walls)-1)]
            width,height = self.get_opening_size(rand_wall)
            #print(f"opening width {width},opening height {height}")
            opening = Opening(rand_wall.wall_transform.x_base,
            rand_wall.wall_transform.y_base,rand_wall.wall_transform.z_base,None,width,height)
            return rand_wall, opening
        elif room.number > 1 and needed_movment_direction == None:
            wall = room.get_wall_at_direction(self.rooms[-1].openings[-1].z_base,180)

            width,height = self.get_opening_size(wall) 
            opening = Opening(wall.wall_transform.x_base,
            wall.wall_transform.y_base,wall.wall_transform.z_base,None,width,height)     
            return wall,opening
        elif room.number > 1 and needed_movment_direction != None:
            wall = room.get_wall_at_direction(direction_vec(needed_movment_direction),0)

            width,height = self.get_opening_size(wall) 
            opening = Opening(wall.wall_transform.x_base,
            wall.wall_transform.y_base,wall.wall_transform.z_base,None,width,height)     
            return wall,opening
        """ else:
            rand_wall = room.non_opening_walls[random.randint(0,len(room.non_opening_walls)-1)]
            width,height = self.get_opening_size(rand_wall)     
            opening = Opening(rand_wall.wall_transform.x_base,
            rand_wall.wall_transform.y_base,rand_wall.wall_transform.z_base,None,width,height)
            return rand_wall, opening   """      

    def get_opening_size(self,wall):
        #return 5,5
        dim = max(self.mep_curve_segmant.width,self.mep_curve_segmant.height)
        width = correct_value(self.opening_size,0,1,2*int(dim),wall.width-1)#random.randint(int(dim)+1,wall.width-1)
        height = correct_value(self.opening_size,0,1,2*int(dim),wall.height-1)#random.randint(int(dim)+1,wall.height-1)
        return width,height

    def get_room_size(self):
        #return 20,20,20, Directions.Positive_z
        if len(self.rooms) == 0:
            needed_movment_direction = Directions(random.randint(0,5))
            x_dim,y_dim,height = self.get_room_dim(needed_movment_direction)
            return x_dim,y_dim,height, needed_movment_direction
        else:
            x_dim = random.randint(2 * ceil(self.mep_curve_segmant.max_elbow_length),self.max_room_dim)
            y_dim = random.randint(2 * ceil(self.mep_curve_segmant.max_elbow_length),self.max_room_dim)
            height = random.randint(2 * ceil(self.mep_curve_segmant.max_elbow_length),self.max_room_dim)
            return x_dim,y_dim,height,None


    def get_room_dim(self,movment_direction):
        x_dim,y_dim,height = None,None,None
        if movment_direction == Directions.Positive_z:
            x_dim = random.randint(ceil(self.mep_curve_segmant.max_dim) + 1,self.max_room_dim)
            y_dim = random.randint(ceil(self.mep_curve_segmant.max_dim)+ 1,self.max_room_dim)
            height = random.randint(ceil(self.mep_curve_segmant.length)*4,self.max_room_dim)
        else:

            min_value = ceil(self.mep_curve_segmant.max_dim) + \
                4 * ceil(self.mep_curve_segmant.max_elbow_length) + 1
            max_value = max(min_value,self.max_room_dim) +1
            x_dim = random.randint(min_value,max_value)

            min_value = ceil(self.mep_curve_segmant.max_dim) + \
                4 * ceil(self.mep_curve_segmant.max_elbow_length) + 1
            max_value = max(min_value,self.max_room_dim) +1
            y_dim = random.randint(min_value,max_value)

            min_value = ceil(self.mep_curve_segmant.max_elbow_length) * 3
            max_value = max(min_value,self.max_room_dim) +1
            

            height = random.randint(min_value,max_value)                

        return x_dim,y_dim,height 

    def generate_room(self,origin,x_dim,y_dim,height,movment_direction,number):

        room = Room(origin,x_dim
        ,y_dim,height,number)

        if room.number > 2:
            wall, opening = self.get_room_opening(self.rooms[-1],movment_direction)
            self.rooms[-1].create_wall_opening(wall, opening)  

        if room.number == 1:
            wall, opening = self.get_room_opening(room,movment_direction)
        else:
            wall, opening = self.get_room_opening(room,None)
    
        room.create_wall_opening(wall, opening)



        if room.number > 1 :
            direction = self.rooms[-1].openings[-1].origin - room.openings[-1].origin
            room.move_room(direction)

            if self.is_clashing(room,self.rooms[-1].walls):
                self.rooms[-1].openings.pop()
                room.delete()
                #print("DIRECTION NOT VALID")
                return False

            common_opening = room.create_common_opening(self.rooms[-1].openings[-1],room.openings[0])
            self.rooms[-1].correct_wall(common_opening,self.rooms[-1].opening_walls[-1])
            room.correct_wall(common_opening,room.opening_walls[-1])

  


        self.rooms.append(room)
        #print("DIRECTION VALID")
        return True
    
    
    def is_clashing(self,room, excluded_elements):
        ignore_elements = []
        ignore_elements.extend(room.walls)
        ignore_elements.extend(excluded_elements)
        for wall in room.walls:
            hit_info = wall.intersects(ignore = tuple(ignore_elements))
            if hit_info:  
                return True

        return False

    def crerate_colliders(self):
        for room in self.rooms:
            for wall in room.walls:
                wall.collider = "mesh"

    def reset(self):
        for room in self.rooms:
            room.delete()
            
        self.rooms = []
        self.path_rects = []
        self.is_last_room = False



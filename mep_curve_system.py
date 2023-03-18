from ursina import *
from utils_ursina import *
from mep_curve_segmant import *
from enums import *
import numpy as np
from transform import Transform
import logging

class System():
    def __init__(self,shape,max_iteration,detection_distance = 30,max_distance_to_goal = 5,radius_mul = 1.5):
        self.shape = shape
        self.radius_mul = radius_mul
        self.detection_distance = detection_distance
        self.max_distance_to_goal = max_distance_to_goal
        self.max_iteration = max_iteration
        self.system_segmants = []
        self.system_connectors = []
        self.depth_map = []
        self.action_mask = []
        self.grounding_rays = []
        self.is_collide = False
        self.is_done = 0
        self.angles = [11.25,22.5,30,45,60,90]
        self.plane_angles = [0, -11.25, -22.5, -30, -45, -60, -90, 11.25, 22.5, 30, 45, 60, 90]
        self.straight_segmant_action = [0.1,0.5,1,5,10]

        self.depth_map_xRange = 270
        self.depth_map_yRange = 270
        self.depth_map_angle_offset = 15
        self.number_of_rays_per_axiss = 3
        self.min_grounding_distance = from_mm_to_ft(300)
        self.max_grounding_distance = from_mm_to_ft(500)
        self.alignment_transform = None

        self.is_create_detection_system = True
        self._is_collision_detection = True
        self.is_dead_end = False


        self.iteration = 0
   
    def set_dim(self,width,height):
        self.width = width/304.8
        self.height = height/304.8

    def set_goal(self, goal):
        self.goal = goal


    def get_angle(self,angle_code):
        if angle_code == 0 or angle_code == 6 or angle_code == 12 or angle_code == 18:
            return 11.25
        elif angle_code == 1 or angle_code == 7 or angle_code == 13 or angle_code == 19:
            return 22.5
        elif angle_code == 2 or angle_code == 8 or angle_code == 14 or angle_code == 20:
            return 30
        elif angle_code == 3 or angle_code == 9 or angle_code == 15 or angle_code == 21:
            return 45
        elif angle_code == 4 or angle_code == 10 or angle_code == 16 or angle_code == 22:
            return 60
        elif angle_code == 5 or angle_code == 11 or angle_code == 17 or angle_code == 23:
            return 90
        else:
            return 0

    def max_iteration_exceeded(self):
        if self.iteration >= self.max_iteration:
            return True
        
        return False

    def is_successful(self,is_new_room):
        if is_new_room:
            return True
        else:
            return False
                
        distance = vec_len(self.relative_direction_to_goal)
        if distance < self.max_distance_to_goal:
            return True
        
        return False

    def create_segmant(self,action_length,action_direction):

        self.iteration+=1

        #print("iteration ", self.iteration)

        if action_direction == 24:
            action_length = 0.1
        elif action_direction == 25:
            action_length = 0.5
        elif action_direction == 26:
            action_length = 1.
        elif action_direction == 27:
            action_length = 5.
        elif action_direction == 28:
            action_length = 10.
        else:
            action_length = 0.01

        if action_length < 0.01:
            action_length = 0.01 

        x_scaler = 0.
        y_scaler = 0.
        z_scaler = 0.
        angle = 0
        movment_direction = MovmentDirection.Z
        new_segmant_length = action_length

        if action_direction >= 0 and action_direction < 6:#positive x negative angle
            angle = self.get_angle(action_direction)
            new_segmant_length_reduction = get_elbow_length(angle,self.width,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length


            x_scaler = new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            movment_direction = MovmentDirection.X

            angle = -angle

        elif action_direction >= 6 and action_direction < 12:#negative x positive angle
            angle = self.get_angle(action_direction)

            new_segmant_length_reduction = get_elbow_length(angle,self.width,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length

            x_scaler = -new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            angle = angle

            movment_direction = MovmentDirection.X

        elif action_direction >= 12 and action_direction < 18:#positive y positive angle
            angle = self.get_angle(action_direction)
            new_segmant_length_reduction = get_elbow_length(angle,self.height,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length

            y_scaler = new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            movment_direction = MovmentDirection.Y


        elif action_direction >= 18 and action_direction < 24:#negative y negative angle
            angle = self.get_angle(action_direction)
            new_segmant_length_reduction = get_elbow_length(angle,self.height,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length

            y_scaler = -new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))


            angle = -angle

            movment_direction = MovmentDirection.Y


        else:
            z_scaler = action_length

            movment_direction = MovmentDirection.Z

        if isinstance(x_scaler,np.float64):
            x_scaler = x_scaler.item()

        if isinstance(y_scaler,np.float64):
            y_scaler = y_scaler.item()

        if isinstance(z_scaler,np.float64):
            z_scaler = z_scaler.item()

        x_translation = x_scaler * self.system_connectors[-1].x_base
        y_translation = y_scaler * self.system_connectors[-1].y_base
        z_translation = z_scaler * self.system_connectors[-1].z_base

        end_position = deepcopy(self.system_segmants[-1].end_position)
        end_position += x_translation
        end_position += y_translation
        end_position += z_translation



        self.current_angle = angle

        if angle == 0:
            if len(self.system_segmants) != 1:
                length = new_segmant_length + self.system_segmants[-1].length
                new_segmant = create_segmant(self.system_segmants[-2],end_position,
                self.system_segmants[-1].angle,self.system_segmants[-1].relative_movement_direction,length,self.radius_mul)
            else:
                length = new_segmant_length + self.system_segmants[-1].length
                new_segmant = create_start_segmant(self.shape,self.width,self.height,length,0,0,self.system_segmants[-1],self.radius_mul)
                  
            destroy(self.system_segmants[-1])
            self.system_segmants.pop()
            self.system_connectors.pop()
            self.system_segmants.append(new_segmant)
            self.system_connectors.append(new_segmant.movment_transform) 
        else: 
            new_segmant = create_segmant(self.system_segmants[-1],end_position,angle,movment_direction,new_segmant_length,self.radius_mul)
            self.system_segmants.append(new_segmant)
            self.system_connectors.append(new_segmant.movment_transform)


        return True

    def get_action_mask(self):
        self.action_mask = []
        dead_end_mask = []
        for i in range(2):
            if i == 0:
                for k in range(2):
                    for j in range(len(self.angles)):

                        new_elbow_length = get_elbow_length(self.angles[j],self.width,self.radius_mul)

                        if self.system_segmants[-1].net_length - new_elbow_length < 0.01:
                            self.action_mask.append(0)
                            dead_end_mask.append(0)
                        else:
                            self.action_mask.append(1)
                            if k == 0:#positive x
                                if new_elbow_length >= self.depth_map[j+13]:
                                    dead_end_mask.append(0)
                                else:
                                    dead_end_mask.append(1)
                            elif k == 1:#negative x
                                if new_elbow_length >= self.depth_map[j+19]:
                                    dead_end_mask.append(0)
                                else:
                                    dead_end_mask.append(1)
            else:
                for k in range(2):
                    for j in range(len(self.angles)):

                        new_elbow_length = get_elbow_length(self.angles[j],self.height,self.radius_mul)

                        if self.system_segmants[-1].net_length - new_elbow_length < 0.01:
                            self.action_mask.append(0)
                            dead_end_mask.append(0)
                        else:
                            self.action_mask.append(1)
                            if k == 0:#positive y
                                if new_elbow_length >= self.depth_map[j+7]:
                                    dead_end_mask.append(0)
                                else:
                                    dead_end_mask.append(1)
                            elif k == 1:#negative y
                                if new_elbow_length >= self.depth_map[j+1]:
                                    dead_end_mask.append(0)
                                else:
                                    dead_end_mask.append(1)

        for straight_action in self.straight_segmant_action:
            self.action_mask.append(1)
            if straight_action >= self.depth_map[0]:
                dead_end_mask.append(0)
            else:
                dead_end_mask.append(1)

        if 1 not in dead_end_mask:
            self.is_dead_end = True

        #self.action_mask.append(1)
        #self.action_mask.append(1)
        #self.action_mask.append(1)
        #self.action_mask.append(1)
        #self.action_mask.append(1)

        return self.action_mask

    def transform_point(self,point, transform):
        
        origin = transform.origin

        vec = point - origin
        vec = [vec.x,vec.y,vec.z]

        x = np.dot(vec , transform.x_base_l)
        y = np.dot(vec , transform.y_base_l)
        z = np.dot(vec , transform.z_base_l)

        new_vec = Vec3(x,y,z)


        return new_vec

    def transform_vec(self,vec, transform):
        
        vec = [vec.x,vec.y,vec.z]

        x = np.dot(vec , transform.x_base_l)
        y = np.dot(vec , transform.y_base_l)
        z = np.dot(vec , transform.z_base_l)

        new_vec = Vec3(x,y,z)


        return new_vec
    def check_for_collision(self):

        if not self._is_collision_detection:
            return False

        if len(self.system_segmants) ==1:
            hit_info = self.system_segmants[-1].intersects()
        else:
            hit_info = self.system_segmants[-1].intersects(ignore = (self.system_segmants[-2],))

        if hit_info.hit:  
            self.is_collide = True      
            return True
    
        self.is_collide = False
        return False

    def create_detection_system(self):

        if not self.is_create_detection_system:
            return 

        depth_map = []#[1]*361
        self.depth_map = depth_map
        
        transform = self.system_connectors[-1]
        origin = transform.origin


        """ depth_map = []

        for y_angle in range(int(-self.depth_map_yRange/2),int(self.depth_map_yRange/2)+1,self.depth_map_angle_offset):
                for x_angle in range(int(-self.depth_map_xRange/2),int(self.depth_map_xRange/2)+1,self.depth_map_angle_offset):
                    direction = self.create_detection_ray_direction(transform,x_angle,y_angle)
                    hit_info = raycast(origin,direction,distance = self.detection_distance,ignore = (self.system_segmants[-1],))
                    depth_map.append(hit_info.distance/self.detection_distance)
                    #line = draw_line(origin,origin + (direction*hit_info.distance))
                    #print(hit_info.distance)
        self.depth_map = depth_map """
        self.grounding_rays = []

        for i in range(2):
            for angle in self.plane_angles:
                if i == 1 and angle == 0:
                    continue

                self.create_detection_plane(transform,angle,RotationAxis(i+1))


    def create_detection_plane(self,transform,angle,rotation_axiss):
        offsest_x = self.width/(self.number_of_rays_per_axiss-1)
        offsest_y = self.height/(self.number_of_rays_per_axiss-1)

        min_distance = self.detection_distance

        x_base = 0
        y_base = 0

        if rotation_axiss == RotationAxis.X:
            x_angle = angle
            y_angle = 0
            x_base = transform.x_base
        else:
            x_angle = 0
            y_angle = angle
            y_base = transform.y_base
            
        direction = self.create_detection_ray_direction(transform,x_angle,y_angle)
        

        if x_base == 0:
            x_base = rotate_Vec_around_vec(y_angle,transform.y_base,transform.x_base)
        if y_base == 0:
            y_base = rotate_Vec_around_vec(x_angle,transform.x_base,transform.y_base)

        new_transform = Transform(x_base,y_base,direction,transform.origin)

        for i in range(self.number_of_rays_per_axiss):
            for j in range(self.number_of_rays_per_axiss):
                new_origin = self.get_new_origin(new_transform,self.width,self.height,i*offsest_x,j*offsest_y)
                hit_info = raycast(new_origin,direction,distance = self.detection_distance,ignore = (self.system_segmants[-1],))
                distance = hit_info.distance
                #line = draw_line(new_origin,new_origin + (direction*distance))
                #print(distance)
                if distance < min_distance:
                    min_distance = distance

        if angle == 90 or angle == -90:
            if rotation_axiss == RotationAxis.X:
                self.grounding_rays.append(min_distance-self.height/2)
            else:
                self.grounding_rays.append(min_distance-self.width/2)


        self.depth_map.append(min_distance)


    def get_new_origin(self,transform,width,height,x_translation,y_translation):

        right = -width/2 *  transform.x_base
        up = -height/2 * transform.y_base

        new_origin = transform.origin + up + right
        new_origin = new_origin + (x_translation * transform.x_base) + (y_translation * transform.y_base)
        
        return new_origin

    def create_detection_ray_direction(self,transform,x_angle,y_angle):
        new_ray_direction = rotate_Vec_around_vec(x_angle,transform.x_base,transform.z_base)
        new_ray_direction = rotate_Vec_around_vec(y_angle,transform.y_base,new_ray_direction)
        return new_ray_direction

    def reset(self):
        for segmant in self.system_segmants:
            destroy(segmant)

        self.system_segmants = []
        self.system_connectors = []     
        self.iteration = 0
        self.is_done = False
        self.is_dead_end = False
        start_segmant = create_start_segmant(self.shape,self.width,self.height,random.random()+0.01,random.random()*90,random.random()*90,radius_mul=self.radius_mul)
        self.system_segmants.append(start_segmant)
        self.system_connectors.append(start_segmant.movment_transform)



    def get_reward(self,is_new_room):
        if self.is_collide:
            reward = -100
            return reward
        else:
            if is_new_room:
                reward = 100

                if self.segmant_rect_angle != 0:
                    reward -= 50

                if self.not_aligned:
                    reward -= 5
                
            else:
                reward = -1

                if self.not_aligned:
                    reward -= 5

                """ if self.current_angle != 0:
                    reward = -1 """

        
        """ if min(self.grounding_rays) < self.min_grounding_distance or min(self.grounding_rays) > self.max_grounding_distance:
            reward -= 5 """


        #print("single agent reward", reward)

        return reward
    
    def get_distance_reward(self,distance):
        return -math.pow(distance/10,0.7)

    def get_status(self,previous_rect,rect,is_new_room):
        
        self.target_vec_for_angle_clac = None
        if(is_new_room):
            self.segmant_rect_angle = angle_between_vec(self.system_connectors[-1].z_base,previous_rect.transform.z_base)
            self.target_vec_for_angle_clac = previous_rect.transform.z_base
        else:
            self.segmant_rect_angle = angle_between_vec(self.system_connectors[-1].z_base,rect.transform.z_base)
            self.target_vec_for_angle_clac = rect.transform.z_base
        
        if self.alignment_transform == None:
            self.alignment_transform = Transform()#the system connectors will be checked for alignment with the y axis of the alignment transform 

        self.angle_with_x_axis = rad_to_deg(angle_between_vec(self.alignment_transform.y_base,self.system_connectors[-1].x_base))
        self.angle_with_y_axis = rad_to_deg(angle_between_vec(self.alignment_transform.y_base,self.system_connectors[-1].y_base))
        self.angle_with_z_axis = rad_to_deg(angle_between_vec(self.alignment_transform.y_base,self.system_connectors[-1].z_base))
 
        if self.angle_with_x_axis != 0 and self.angle_with_x_axis != 180 and self.angle_with_y_axis != 0 and self.angle_with_y_axis != 180 and self.angle_with_z_axis != 0 and self.angle_with_z_axis != 180:
            self.not_aligned = True
        else:
            self.not_aligned = False
        
        self.min_rect_point = self.transform_point(rect.min,self.system_connectors[-1])
        self.max_rect_point = self.transform_point(rect.max,self.system_connectors[-1])

        """ x_base = self.system_connectors[-1].x_base
        y_base = self.system_connectors[-1].y_base
        z_base = self.system_connectors[-1].z_base """

        state = []

        state.extend(self.depth_map)#361 + 25 = 386

        """ state.extend(vec3_to_list(x_base))#3
        state.extend(vec3_to_list(y_base))#3
        state.extend(vec3_to_list(z_base))#3 """
        state.append(self.system_segmants[-1].net_length)#1

        state.extend(vec3_to_list(self.min_rect_point))#3
        state.extend(vec3_to_list(self.max_rect_point))#3


        state.append(self.width)#1
        state.append(self.height)#1

        self.target_vec_for_angle_clac = self.transform_vec(self.target_vec_for_angle_clac,self.system_connectors[-1])
        state.extend(vec3_to_list(self.target_vec_for_angle_clac))#38
        
        self.alignment_vec = self.transform_vec(self.alignment_transform.y_base,self.system_connectors[-1])
        state.extend(vec3_to_list(self.alignment_vec))#41


        return state


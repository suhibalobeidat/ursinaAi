from ursina import *
from utils_ursina import *
from mep_curve_segmant import *
from enums import *
import numpy as np
from transform import Transform

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
        self.is_collide = False
        self.is_done = True
        self.angles = [11.25,22.5,30,45,60,90]
        self.plane_angles = [0, -11.25, -22.5, -30, -45, -60, -90, 11.25, 22.5, 30, 45, 60, 90]
        
        self.depth_map_xRange = 270
        self.depth_map_yRange = 270
        self.depth_map_angle_offset = 15
        self.number_of_rays_per_axiss = 3

        self.is_create_detection_system = True
        self._is_collision_detection = True
   
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

    def is_successful(self):
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
            action_length = 1
        elif action_direction == 27:
            action_length = 5
        elif action_direction == 28:
            action_length = 10
        else:
            action_length = 0.01

        if action_length < 0.01:
            action_length = 0.01 

        x_scaler = 0
        y_scaler = 0
        z_scaler = 0
        angle = 0
        movment_direction = MovmentDirection.Z
        new_segmant_length = action_length

        if action_direction >= 0 and action_direction < 6:
            angle = self.get_angle(action_direction)
            new_segmant_length_reduction = get_elbow_length(angle,self.width,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length


            x_scaler = new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            movment_direction = MovmentDirection.X

            angle = -angle

        elif action_direction >= 6 and action_direction < 12:
            angle = self.get_angle(action_direction)

            new_segmant_length_reduction = get_elbow_length(angle,self.width,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length

            x_scaler = -new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            angle = angle

            movment_direction = MovmentDirection.X

        elif action_direction >= 12 and action_direction < 18:
            angle = self.get_angle(action_direction)
            new_segmant_length_reduction = get_elbow_length(angle,self.height,self.radius_mul)
            new_segmant_length = new_segmant_length_reduction + action_length

            y_scaler = new_segmant_length * np.sin(deg_to_rad(angle))
            z_scaler = new_segmant_length * np.cos(deg_to_rad(angle))

            movment_direction = MovmentDirection.Y


        elif action_direction >= 18 and action_direction < 24:
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
                new_segmant = create_start_segmant(self.shape,self.width,self.height,length,self.radius_mul)
                  
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
        for i in range(2):
            if i == 0:
                for k in range(2):
                    for j in range(6):
                        if self.system_segmants[-1].net_length - get_elbow_length(self.angles[j],self.width,self.radius_mul) < 0.01:
                            self.action_mask.append(0)
                        else:
                            self.action_mask.append(1)
            else:
                for k in range(2):
                    for j in range(6):
                        if self.system_segmants[-1].net_length - get_elbow_length(self.angles[j],self.height,self.radius_mul) < 0.01:
                            self.action_mask.append(0)
                        else:
                            self.action_mask.append(1)

        self.action_mask.append(1)
        self.action_mask.append(1)
        self.action_mask.append(1)
        self.action_mask.append(1)
        self.action_mask.append(1)

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

        """ l = [1]*386
        self.depth_map = l
        return l """
        
        transform = self.system_connectors[-1]
        origin = transform.origin


        depth_map = []

        for y_angle in range(int(-self.depth_map_yRange/2),int(self.depth_map_yRange/2)+1,self.depth_map_angle_offset):
                for x_angle in range(int(-self.depth_map_xRange/2),int(self.depth_map_xRange/2)+1,self.depth_map_angle_offset):
                    direction = self.create_detection_ray_direction(transform,x_angle,y_angle)
                    hit_info = raycast(origin,direction,distance = self.detection_distance,ignore = (self.system_segmants[-1],))
                    depth_map.append(hit_info.distance/self.detection_distance)
                    #line = draw_line(origin,origin + (direction*hit_info.distance))
                    #print(hit_info.distance)
        self.depth_map = depth_map

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

        self.depth_map.append(min_distance/self.detection_distance)


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
        start_segmant = create_start_segmant(self.shape,self.width,self.height,1,self.radius_mul)
        self.system_segmants.append(start_segmant)
        self.system_connectors.append(start_segmant.movment_transform)



    def get_reward(self,is_new_room):
        if self.is_collide:
            reward = -100
        else:
            distance = vec_len(self.relative_direction_to_goal)
            if is_new_room:
                reward = 50

            else:
                reward = self.get_distance_reward(distance)
   
                if self.current_angle == 0:
                    reward = reward/100

        #print("single agent reward", reward)

        return reward
    
    def get_distance_reward(self,distance):
        return -math.pow(distance/10,0.7)

    def get_status(self,rect):

 
        self.relative_direction_to_goal = self.transform_point(rect.origin,self.system_connectors[-1])
        self.min_rect_point = self.transform_point(rect.min,self.system_connectors[-1])
        self.max_rect_point = self.transform_point(rect.max,self.system_connectors[-1])

        x_base = self.system_connectors[-1].x_base
        y_base = self.system_connectors[-1].y_base
        z_base = self.system_connectors[-1].z_base

        state = []

        state.extend(self.depth_map)#361 + 25 = 386

        state.extend(vec3_to_list(x_base))#3
        state.extend(vec3_to_list(y_base))#3
        state.extend(vec3_to_list(z_base))#3
        state.append(self.system_segmants[-1].net_length)#1

        state.extend(vec3_to_list(self.min_rect_point))#3
        state.extend(vec3_to_list(self.max_rect_point))#3


        state.append(self.width)#1
        state.append(self.height)#1

        if len(self.system_segmants) == 1:
            state.append(self.system_segmants[-1].length)#1
            state.append(0)#1
            state.extend(vec3_to_list(z_base))#3
            state.append(0)#1
            state.extend(vec3_to_list(z_base))#3
        elif len(self.system_segmants) == 2:
            state.append(self.system_segmants[-1].length)
            state.append(self.system_segmants[-2].length)
            state.extend(vec3_to_list(self.system_connectors[-2].z_base)) 
            state.append(0)
            state.extend(vec3_to_list(self.system_connectors[-2].z_base)) 
        else:
            state.append(self.system_segmants[-1].length)
            state.append(self.system_segmants[-2].length)
            state.extend(vec3_to_list(self.system_connectors[-2].z_base)) 
            state.append(self.system_segmants[-3].length)
            state.extend(vec3_to_list(self.system_connectors[-3].z_base))   

        return state

    """ def get_reward(self):
        if self.is_collide:
            reward = -100
        else:
            distance = vec_len(self.relative_direction_to_goal)
            if distance < self.max_distance_to_goal:
                reward = 100

            else:
                reward = self.get_distance_reward(distance)
   
                if self.current_angle == 0:
                    reward = reward/100

        #print("single agent reward", reward)

        return reward """
    


    """ def get_status(self):

 
        self.relative_direction_to_goal = self.transform_point(self.goal,self.system_connectors[-1])

        x_base = self.system_connectors[-1].x_base
        y_base = self.system_connectors[-1].y_base
        z_base = self.system_connectors[-1].z_base

        state = []

        state.extend(self.depth_map)

        state.extend(vec3_to_list(x_base))
        state.extend(vec3_to_list(y_base))
        state.extend(vec3_to_list(z_base))

        state.append(self.system_segmants[-1].net_length)

        state.extend(vec3_to_list(self.relative_direction_to_goal.normalized()))
        state.append(vec3_len(self.relative_direction_to_goal))

        state.append(self.width)
        state.append(self.height)

        #if len(self.system_segmants) == 1:
        #    state.append(0)
        #    state.extend(vec3_to_list(z_base))
        #else:
        #    state.append(self.system_segmants[-2].length)
        #    state.extend(vec3_to_list(self.system_connectors[-2].z_base)) 

        return state """
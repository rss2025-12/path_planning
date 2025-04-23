import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler


assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Quaternion, Point, Pose
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
# from tf_transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R

from scipy.ndimage import grey_dilation
import numpy as np
import heapq

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_debug_pub = self.create_publisher(
            OccupancyGrid,
            "/map_debug",
            1
        )

        self.point_debug_pub = self.create_publisher(PoseArray, "/debug_points", 1)


        # Map vars
        self.map = None
        self.resolution = None
        self.disk_radius = 10 # In pixels

        # Pose vars
        self.initial_pose = None
        self.goal_pose = None

        # Planning vars
        self.search_based = False # False: sample_based

        self.get_logger().info("Path planner initialized")

    def map_cb(self, msg):
        """
        Receives a map, and transforms it into a 2d np array
        with 0 as free space and 1 as occupied, the original
        map's occupied spaces are dialated by self.disk_radius
        """
        width = msg.info.width
        height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_pos = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.origin_ori = msg.info.origin.orientation
        quat = [self.origin_ori.x,
                self.origin_ori.y,
                self.origin_ori.z,
                self.origin_ori.w]
        _, _, theta = R.from_quat(quat).as_euler('xyz', degrees=False)
        self.T_world_map = np.array([[np.cos(theta), -np.sin(theta), self.origin_pos[0]],
                                     [np.sin(theta), np.cos(theta), self.origin_pos[1]],
                                     [0, 0, 1]])
        self.T_map_world = np.linalg.inv(self.T_world_map)

        data = np.array(msg.data).reshape((height, width))

        # Disk dialation
        obstacle_map = np.isin(data, [100, -1]).astype(np.uint8)
        disk_radius = np.ceil(self.disk_radius)
        y, x = np.ogrid[-disk_radius:disk_radius+1,
                        -disk_radius:disk_radius+1]
        disk = x**2 + y**2 <= disk_radius**2
        self.map = grey_dilation(obstacle_map, footprint=disk)

        # Debugging
        map_for_debug = np.where(self.map == 1, 100, 0)
        msg.data = map_for_debug.flatten().tolist()
        self.map_debug_pub.publish(msg)

        self.get_logger().info(f"Map recieved")

    def pose_cb(self, pose):
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y
        quaternion = [pose.pose.pose.orientation.x,
                    pose.pose.pose.orientation.y,
                    pose.pose.pose.orientation.z,
                    pose.pose.pose.orientation.w]
        _, _, theta = R.from_quat(quaternion).as_euler('xyz', degrees=False)

        self.initial_pose = np.array([x, y, theta])
        self.get_logger().info(f"Initial Pose Received")
        self.try_plan_path()

    def goal_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        quaternion = [msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w]
        _, _, theta = R.from_quat(quaternion).as_euler('xyz', degrees=False)

        self.goal_pose = np.array([x, y, theta])
        self.get_logger().info(f"Goal Pose Received")
        self.try_plan_path()

    def try_plan_path(self):
        if self.map is not None and self.initial_pose is not None and self.goal_pose is not None:
            self.plan_path(self.initial_pose, self.goal_pose, self.map)
            self.initial_pose = None
            self.goal_pose = None

    def plan_path(self, start_point, end_point, map):
        if self.search_based:
            pixel_path = self.a_star(start_point, end_point, map)
        else:
            pixel_path = self.rrt(start_point, end_point, goal_radius=3)

        if pixel_path is None or len(pixel_path) == 0:
            self.get_logger().warn("No path found")
            return

        self.trajectory.clear()
        for mx, my in pixel_path:
            wx, wy = self.map_to_world(mx, my)
            self.trajectory.addPoint((wx, wy))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def world_to_map(self, x, y):
        T_world_point = np.array([[1, 0, x],
                                  [0, 1, y],
                                  [0, 0, 1]])

        T_map_point = self.T_map_world @ T_world_point

        mx = int(T_map_point[0][2] / self.resolution)
        my = int(T_map_point[1][2] / self.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        mx = mx * self.resolution
        my = my * self.resolution

        T_map_point = np.array([[1, 0, mx],
                                [0, 1, my],
                                [0, 0, 1]])

        T_world_point = self.T_world_map @ T_map_point
        return T_world_point[0][2], T_world_point[1][2]

    def a_star(self, start_point, end_point, map):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))  # Euclidean

        def reconstruct_path(parent, current):
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        height, width = map.shape

        start = self.world_to_map(start_point[0], start_point[1])
        goal = self.world_to_map(end_point[0], end_point[1])

        self.get_logger().info(f"{start=}")
        self.get_logger().info(f"{goal=}")

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # (f_score, g_score, node)

        parent = {}
        g_score = {start: 0}
        visited = set()

        directions = [  # 8-connected grid
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return reconstruct_path(parent, current)

            if current in visited:
                continue

            visited.add(current)
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue  # Skip out-of-bounds

                if map[neighbor[1], neighbor[0]] != 0:
                    continue  # Skip obstacles

                tentative_g = g_score[current] + np.linalg.norm([dx, dy])

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    parent[neighbor] = current

        return []

    # RRT methods
    def sample(self, buffer_ratio=5):
        map_height, map_width = self.map.shape
        ###
        rand_x = np.random.uniform(0, map_width)
        rand_y = np.random.uniform(0, map_height)

        return (rand_x, rand_y)
        ###

        ###
        # xl, xh = self.rrt_xbounds[0]-sample_buffer, self.rrt_xbounds[1]+sample_buffer
        # yl, yh = self.rrt_ybounds[0]-sample_buffer, self.rrt_ybounds[1]+sample_buffer

        # xl = np.clip(xl, a_min=0, a_max=map_width)
        # xh = np.clip(xh, a_min=0, a_max=map_width)
        # yl = np.clip(yl, a_min=0, a_max=map_height)
        # yh = np.clip(yh, a_min=0, a_max=map_height)

        # x_buffer = buffer_ratio * abs(self.rrt_xbounds[1] - self.rrt_xbounds[0] + 5)
        # y_buffer = buffer_ratio * abs(self.rrt_ybounds[1] - self.rrt_xbounds[0] + 5)
        
        # xl = np.max(np.array([self.rrt_xbounds[0] - x_buffer, 0])) 
        # xh = np.min(np.array([self.rrt_xbounds[1] + x_buffer, map_width]))
        # yl = np.max(np.array([self.rrt_ybounds[0] - y_buffer, 0]))
        # yh = np.min(np.array([self.rrt_ybounds[1] + y_buffer, map_height]))

        # rand_x = np.random.uniform(xl, xh)
        # rand_y = np.random.uniform(yl, yh)

        # rand_theta = np.random.uniform(0, 2*np.pi)
        # return (rand_x, rand_y)
        ###

    # def nearest(self, curr_point, rand_nodes, thresh_ang=np.pi/2):
    #     # rand_points is an np arrays of tuples representing points
    #     # curr_point is our current starting position
    #     rand_points = [(rand_node.value[:2], rand_node.value[2]) for rand_node in rand_nodes]
        
    #     distances = []
    #     for point, point_ang in rand_points:
    #         ang = np.arctan2(*(curr_point-point)[::-1]) # arctan(y, x)
    #         if np.abs(point_ang-ang) < thresh_ang:
    #             distances.append(np.linalg.norm(np.array(curr_point) - np.array(point)))

    #     return rand_nodes[np.argmin(np.array(distances))] #, axis=0

    def nearest(self, curr_point, rand_nodes, thresh_ang=np.pi/2):
        # min_dist = float('inf')
        min_dist_no_ang = float('inf')
        # nearest_node = None
        nearest_node_no_ang = None

        for rand_node in rand_nodes:
            point = rand_node.value[:2]
            # point_ang = rand_node.value[2]

            # angle_to_point = np.arctan2(*(curr_point - point)[::-1])  # arctan(y, x)
            dist = np.linalg.norm(curr_point - point)
            if dist < min_dist_no_ang:
                min_dist_no_ang = dist
                nearest_node_no_ang = rand_node
                # if (np.abs(point_ang - angle_to_point) < thresh_ang) and (min_dist > dist):
                #     min_dist = dist
                #     nearest_node = rand_node
                    # dist = np.linalg.norm(curr_point - point)
                
        # if nearest_node: 
        #     return nearest_node
        # else: 
        return nearest_node_no_ang
        # return nearest_node


    # def steer(self, znearest, zrand, max_steer_ang=3*np.pi/4, L=0.5, lookahead=1):
    #     # include dynamic and kinematic constraints?
    #     # motion model?

    #     # need to limit max steering angle, will find necessary angle using pure pursuit model
    #     # map frame
    #     # TODO: CONVERT POINTS FROM MAP FRAME TO ROBOT FRAME
    #     # robot frame:
    #     x_robot, y_robot, theta_robot = znearest
    #     x_wp, y_wp = zrand
    #     dx = x_wp - x_robot
    #     dy = y_wp - y_robot

    #     if (dx**2+dy**2)**(0.5) > lookahead:
    #         y = lookahead/np.sqrt(1+(dx/dy)**2)
    #         x = y*(dx/dy)
    #         dx, dy = x, y

    #     angle_to_wp = np.arctan2(dy, dx)

    #     # Calculate the steering angle (geometry of the pursuit)
    #     angle_diff = angle_to_wp
    #     # Pure Pursuit formula for steering angle (in radians)
    #     req_steer_ang = np.arctan2(2.0* L * np.sin(angle_diff), lookahead)
    #     if np.abs(req_steer_ang) > max_steer_ang:
    #         steer_angle = (req_steer_ang/np.abs(req_steer_ang))*max_steer_ang
    #     else:
    #         steer_angle = req_steer_ang

    #     R = lookahead/(2*np.sin(steer_angle))
    #     alpha = np.arcsin(lookahead*np.tan(steer_angle)/(2*L))
    #     gamma1 = 2*alpha
    #     dtheta = gamma1

    #     x_new, y_new, theta_new = 0,0,0

    #     min_dist = None
    #     x_out, y_out, theta_out = None

    #     while theta_new <= 2*np.pi:
    #         dy_i = R*(np.cos(dtheta+theta_new)-np.cos(theta_new))
    #         dx_i = R*(np.sin(dtheta+theta_new)-np.sin(theta_new))
    #         theta_new = dtheta + theta_new
    #         x_new, y_new = x_new + dx_i, y_new + dy_i
    #         goal_dist = np.linalg.norm(np.array([x_wp-x_new, y_wp-y_new]))
    #         if min_dist == None or goal_dist < min_dist:
    #             min_dist = goal_dist
    #             x_out, y_out, theta_out = x_new, y_new, theta_new

    #     return np.array(x_out, y_out, theta_out)

    def steer(self, znearest, zrand, max_steer_ang=3*np.pi/4, L=0.5, lookahead=1.0, dd=0.1):
        x_robot, y_robot, theta_robot = znearest
        x_wp, y_wp = zrand

        # Transform waypoint into robot frame
        dx = x_wp - x_robot # map frame
        dy = y_wp - y_robot # map frame
        dx_r = np.cos(-theta_robot) * dx - np.sin(-theta_robot) * dy
        dy_r = np.sin(-theta_robot) * dx + np.cos(-theta_robot) * dy
        x = dx_r
        y = dy_r

        dist = np.hypot(dx_r, dy_r)
        if dist > lookahead:
            ### OLD >>>
            dx_r = lookahead / np.sqrt(1 + (x / y) ** 2)
            dy_r = dx_r * (y / x)
            ### OLD <<<
            # self.get_logger().info(f'point {(x, y)} too far, now {(dx_r, dy_r)}')
            # ### NEW >>>
            # x = dx_r
            # y = dy_r
            # while self.in_collision(np.array(x, y)) and np.abs(dx_r/dy_r*dd) < np.abs(x) and np.abs(dy_r/dx_r*dd) < np.abs(y):
            #         x -= dx_r/dy_r*dd
            #         y -= dy_r/dx_r*dd
            # dx_r = x
            # dy_r = y
            # ### NEW <<<
        
        # works? until here

        angle_to_wp = np.arctan2(dy_r, dx_r)

        # TODO: convert old code for find point to new to invert pure pursuit model

        ###
        # Pure Pursuit steering angle
        req_steer_ang = np.arctan2(2 * L * np.sin(angle_to_wp), lookahead) #suspicious #imposter

        # Clip to max steering angle
        steer_angle = np.clip(req_steer_ang, -max_steer_ang, max_steer_ang)

        ### OLD FIND POINT >>>
        # # Compute turning radius and delta heading
        # R = L / np.tan(steer_angle) #suspicious
        # dtheta = lookahead / R #suspicious

        # # Compute new pose in robot frame
        # x_new = R * np.sin(dtheta)
        # y_new = R * (1 - np.cos(dtheta))
        # theta_new = dtheta
        ### <<< OLD FIND POINT

        ###

        ###
        # x_new = dx_r
        # y_new = dy_r
        # theta_new = angle_to_wp
        ###

        # ### NEW FIND POINT >>>
        theta_new = np.arcsin(lookahead*np.tan(steer_angle)/(2*L))
        x_new = lookahead*np.cos(theta_new)
        y_new = lookahead*np.sin(theta_new)

        # ### <<< NEW FIND POINT

        # Transform back to map frame
        x_map = x_robot + np.cos(theta_robot) * x_new - np.sin(theta_robot) * y_new
        y_map = y_robot + np.sin(theta_robot) * x_new + np.cos(theta_robot) * y_new
        theta_map = (theta_robot + theta_new) % (2 * np.pi)

        # x = x_wp - x_robot
        # y = y_wp - y_robot
        # dx_r = x
        # dy_r = y

        # dist = np.hypot(x, y)
        # if dist > lookahead:
        #     dx_r = lookahead / np.sqrt(1 + (x / y) ** 2)
        #     dy_r = dx_r * (y / x)

        # x_map = x_robot + dx_r
        # y_map = y_robot + dy_r
        # theta_map = np.arctan2(y,x)

        return np.array([x_map, y_map, theta_map])

    def in_collision(self, point, debug=False):
        # check that point is not in occupancy grid or out of bounds
        map_height, map_width = self.map.shape
        if debug: self.get_logger().info(f'using bounds w = {map_width}, h = {map_height} for collision check')

        if not (0 <= point[0] < map_width and 0 <= point[1] < map_height):
            if debug: self.get_logger().info(f'pt outside bounds')
            return True

        ### OLD >>>
        # if self.map[int(point[1]), int(point[0])] != 0:
        #     if debug: self.get_logger().info(f'pt inside bounds, not free')
        #     return True
        ### OLD <<<
        
        ### NEW >>>
        round_pt = np.round(point)
        # self.get_logger().info(f'the round pt is {round_pt}')
        if self.map[int(round_pt[1]), int(round_pt[0])] != 0:
            if debug: self.get_logger().info(f'pt inside bounds, not free')
            return True
        ### NEW <<<
        
        return False

    def collision_free(self, xcurrent, xnew, steps = 1000):
        # check that all points along path from xcurrent to xnew are not in collision
        direction = xnew - xcurrent
        step = direction/steps
        for i in range(steps):
            pt = xcurrent + step
            if self.in_collision(pt):
                return False
        return True
    
    def generate_hall_dist(self, step=0.1):
        map_height, map_width = self.map.shape
        
        # self.get_logger().info(f' the sample range is w = {map_width}, h = {map_height}')
        xrange = np.arange(0, map_width-1, step=step)
        yrange = np.arange(0, map_height-1, step=step)
        # self.get_logger().info(f' the sample range is x = {xrange[-5:]}, y = {yrange[-5:]}')
        # self.get_logger().info(f' map is {self.map}')

        samples = set()
        for x in xrange:
            for y in yrange:
                pt = (x,y)
                # self.get_logger().info(f'pt is {pt}')
                if not self.in_collision(pt):
                    samples.add(pt)
        # self.get_logger().info(f'the map is shape {self.map.shape} and the map is {self.map}')
        # self.get_logger().info(f'the generated possible samples are {np.array(samples)}')
        return np.array(list(samples))
    
    def sample_free(self, samples):
        l = len(samples)
        return samples[np.random.choice(l)]
    
    def sample_resample(self):
        map_height, map_width = self.map.shape
        pt = (-1,-1)
        while self.in_collision(pt):
            rand_x = np.random.uniform(0, map_width-1)
            rand_y = np.random.uniform(0, map_height-1)
            pt = (rand_x, rand_y)

        return pt


    def rrt(self, start_point_map, end_point_map, goal_radius=1, sample_buffer=2, buffer_ratio = 1/3):
        """
        all points are in the map frame
        """
        start_time = self.get_clock().now().nanoseconds
        map_width, map_height = self.map.shape
        loop_cap = 10000
        # start_point = (self.world_to_map(*start_point_map[:2])[0], self.world_to_map(*start_point_map[:2])[1], [start_point_map[2]-np.pi])
        # end_point = (self.world_to_map(*end_point_map[:2])[0], self.world_to_map(*end_point_map[:2])[1], end_point_map[2]-np.pi)
        start_point = np.array([*self.world_to_map(*start_point_map[:2]), start_point_map[2]-np.pi])
        end_point = np.array([*self.world_to_map(*end_point_map[:2]), end_point_map[2]-np.pi])
        self.get_logger().info(f'the goal point in the map frame is {end_point} and in collision = {self.in_collision(end_point[:2], debug=True)}')

        nodes = [RRTNode(start_point)]
        self.rrt_xbounds = np.array([start_point[0], start_point[0]])
        self.rrt_ybounds = np.array([start_point[1], start_point[1]])
        # self.get_logger().info(f'xbounds are {self.rrt_xbounds}')

        # ### NEW 2 >>>
        # self.free_samples = self.generate_hall_dist(step=1)
        # ### NEW 2 <<<

        for i in range(loop_cap):
            # self.get_logger().info(f'the iteration is {i} mickey mice')
            if i % 20 == 0: # OLD: 3 # NEW: 20
                random_point = end_point[:2]
            else: 
                # ### OLD >>>
                # random_point = self.sample(buffer_ratio=buffer_ratio) # map frame
                # ### OLD <<<
                ### NEW
                # if self.in_collision(random_point) or np.linalg.norm(np.array(random_point) - self.nearest(random_point, nodes).value[:2]) > 5:
                #     i -= 1
                #     continue
                ### NEW
                # ### NEW 2 >>>
                # random_point = self.sample_free(self.free_samples)
                # ### NEW 2 <<<
                ### NEW 3 >>>
                random_point = self.sample_resample()
                ### NEW 3 <<<
            parent = self.nearest(random_point, nodes) # map frame
            new_point = self.steer(parent.value, random_point, lookahead=4, max_steer_ang=np.pi/4) # OLD
            # new_point = self.steer(parent.value, random_point, lookahead=1.5, max_steer_ang=np.pi/10) # NEW
            # self.get_logger().info(f'adding new point {new_point} with parent {parent.value[:2]}')
            # if not self.collision_free(parent.value, new_point) and parent.parent:
            #     parent = parent.parent
            #     new_point = self.steer(parent.value, random_point, lookahead=1.)

            # self.get_logger().info(f'xbounds are {self.rrt_xbounds}')
            if self.collision_free(parent.value, new_point):
                # self.get_logger().info(f'not in collision, adding point {new_point}')
                new_node = RRTNode(new_point)
                new_node.parent = parent
                nodes.append(new_node)

                ##
                # if new_point[0] < self.rrt_xbounds[0]:
                #     self.rrt_xbounds[0] = np.max(np.array([self.rrt_xbounds[0] - sample_buffer, 0])) 
                # elif new_point[0] > self.rrt_xbounds[1]:
                #     self.rrt_xbounds[1] = np.min(np.array([self.rrt_xbounds[1] + sample_buffer, map_width]))
                # # self.get_logger().info(f'xbounds are {self.rrt_xbounds}')


                # if new_point[1] < self.rrt_ybounds[0]:
                #     self.rrt_ybounds[0] = np.max(np.array([self.rrt_ybounds[0] - sample_buffer, 0]))
                # elif new_point[1] > self.rrt_ybounds[1]:
                #     self.rrt_ybounds[1] = np.min(np.array([self.rrt_ybounds[1] + sample_buffer, map_height]))
                ##

                if np.linalg.norm(new_node.value[:2] - end_point[:2]) <= goal_radius: 
                    end_time = self.get_clock().now().nanoseconds
                    self.get_logger().info(f'found goal; final time = {(end_time - start_time)/1e9}')
                    # self.get_logger().info(f'within goal radius')
                    path = []
                    curr_node = new_node
                    while curr_node.parent:
                        path.append(curr_node.value[:2])
                        curr_node = curr_node.parent
                    return path[::-1]
            else:
                # self.get_logger().info(f'in collision')
                pass

            if i%500 == 0:
                # self.get_logger().info(f'{[node.value for node in nodes]}')
                pass
            if nodes:
                self.pub_rrt_pt(np.array(nodes))

        return None

    def pub_rrt_pt(self, nodes):
        # self.get_logger().info(f'trying to publish {[node.value for node in nodes]}')
        points = [self.map_to_world(*node.value[:2]) for node in nodes]
        viz_points = PoseArray()
        viz_points.header.stamp = self.get_clock().now().to_msg()
        viz_points.header.frame_id = self.map_topic

        viz_points.poses = []
        for i, point in enumerate(points):
            q = quaternion_from_euler(0, 0, nodes[i].value[2]+np.pi)
            quaternion_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            pose = Pose(
                position=Point(x=point[0], y=point[1], z=0.0),
                orientation=quaternion_msg
            )
            viz_points.poses.append(pose)

        self.point_debug_pub.publish(viz_points)

class RRTNode:
    def __init__(self, value):
        self.value = value
        self.parent = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        self.children.remove(child_node)

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

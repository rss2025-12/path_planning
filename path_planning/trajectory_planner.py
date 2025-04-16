import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
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
            pixel_path = self.rrt(start_point, end_point)

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
    def sample(self):
        map_width, map_height = self.map.shape
        rand_x = np.random.randint(0, map_width)
        rand_y = np.random.randint(0, map_height)
        # rand_theta = np.random.uniform(0, 2*np.pi)
        return (rand_x, rand_y)

    def nearest(self, curr_point, rand_points):
        # rand_points is an np arrays of tuples representing points
        # curr_point is our current starting position
        distances = []
        for node in rand_points:
            distances.append(np.linalg.norm(np.array(curr_point) - np.array(node.value[:2])))

        return rand_points[np.argmin(np.array(distances), axis=0)]

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

    def steer(self, znearest, zrand, max_steer_ang=3*np.pi/4, L=0.5, lookahead=1.0):
        x_robot, y_robot, theta_robot = znearest
        x_wp, y_wp = zrand

        # Transform waypoint into robot frame
        dx = x_wp - x_robot
        dy = y_wp - y_robot
        dx_r = np.cos(-theta_robot) * dx - np.sin(-theta_robot) * dy
        dy_r = np.sin(-theta_robot) * dx + np.cos(-theta_robot) * dy

        dist = np.hypot(dx_r, dy_r)
        if dist > lookahead:
            dy_r = lookahead / np.sqrt(1 + (dx_r / dy_r) ** 2)
            dx_r = dy_r * (dx_r / dy_r)

        angle_to_wp = np.arctan2(dy_r, dx_r)

        # Pure Pursuit steering angle
        req_steer_ang = np.arctan2(2 * L * np.sin(angle_to_wp), lookahead)

        # Clip to max steering angle
        steer_angle = np.clip(req_steer_ang, -max_steer_ang, max_steer_ang)

        # Compute turning radius and delta heading
        R = L / np.tan(steer_angle)
        dtheta = lookahead / R

        # Compute new pose in robot frame
        x_new = R * np.sin(dtheta)
        y_new = R * (1 - np.cos(dtheta))
        theta_new = dtheta

        # Transform back to map frame
        x_map = x_robot + np.cos(theta_robot) * x_new - np.sin(theta_robot) * y_new
        y_map = y_robot + np.sin(theta_robot) * x_new + np.cos(theta_robot) * y_new
        theta_map = (theta_robot + theta_new) % (2 * np.pi)

        return np.array([x_map, y_map, theta_map])

    def in_collision(self, x):
        # check that point is not in occupancy grid or out of bounds
        map_width, map_height = self.map.shape
        if not (0 <= x[0] < map_width and 0 <= x[1] < map_height):
            return False

        if self.map[x[1], x[0]] != 0:
            return False
        return True

    def collision_free(self, xcurrent, xnew, steps = 1000):
        # check that all points along path from xcurrent to xnew are not in collision
        direction = xnew - xcurrent
        step = direction/steps
        for i in range(steps):
            pt = xcurrent + step
            if self.in_collision(pt):
                return False
        return True

    def rrt(self, start_point, end_point, goal_radius=1):
        loop_cap = 10000
        nodes = [RRTNode(start_point)]
        for i in range(loop_cap):
            self.get_logger().info(f'the iteration is {i} mickey mice')
            random_point = self.sample()
            parent = self.nearest(random_point, nodes)
            new_point = self.steer(parent.value, random_point)
            if not self.in_collision(new_point) and self.collision_free(parent.value, new_point):
                new_node = RRTNode(new_point)
                new_node.parent = parent
                nodes.append(new_node)

            if np.linalg.norm(new_node.value - end_point) <= goal_radius:
                path = []
                curr_node = new_node
                while curr_node.parent:
                    path.append(curr_node.value)
                    curr_node = curr_node.parent
                return path
        return None

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

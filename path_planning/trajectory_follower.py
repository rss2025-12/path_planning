import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from .utils import LineTrajectory
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Bool
import numpy as np

class PurePursuit(Node):
    """
    Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.get_logger().info(f'Pubishing commands to drive topic: {self.drive_topic}')
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.intersect_pub = self.create_publisher(Marker,
                                                   "/intersection",
                                                   1)
        self.reverse_sub = self.create_subscription(Bool, 
                                                    '/back_up', 
                                                    self.reverse_cb, 
                                                    1)

        self.declare_parameter('drive_speed', 1.0)

        self.speed = 1.0 # self.get_parameter('drive_speed').get_parameter_value().double_value
        self.lookahead = 1.0 #0.5 #2.0  # 2.25 * self.speed**2
        self.min_lookahead = 1.5
        self.max_lookahead = 4.0
        self.lookahead_threshold = 2.5
        self.wheelbase_length = 0.1

        self.get_logger().info(f"Speed is {self.speed}")

        self.trajectory = LineTrajectory(node=self, viz_namespace="/followed_trajectory")
        self.initialized_traj = False
        self.progress_index = 0
        self.simplify_traj = True

        self.visualize_intersect = True
        self.reverse = False

        self.get_logger().info("Path follower initialized")

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory with {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)

        if self.simplify_traj is True:
            new_points = self.remove_colinear_points(self.trajectory.points)
            self.trajectory.clear()

            for point in new_points:
                self.trajectory.addPoint(point)

        self.trajectory.publish_viz(duration=0.0)
        self.points = np.array(self.trajectory.points)
        self.get_logger().info(f"Added new trajectory with {len(self.points)} points")
        self.initialized_traj = True

    def remove_colinear_points(self, trajectory, tol=0.0018):
        """
        Removes colinear intermediate points from a 2D trajectory.

        Parameters:
            trajectory: list of (x, y) tuples
            tol: tolerance for considering cross product as zero (float)

        Returns:
            simplified_trajectory: list of (x, y) tuples
        """
        traj = np.array(trajectory)
        if len(traj) <= 2:
            return trajectory  # Nothing to simplify

        simplified = [traj[0]]
        for i in range(1, len(traj) - 1):
            a = traj[i - 1]
            b = traj[i]
            c = traj[i + 1]

            v1 = b - a
            v2 = c - b
            cross = v1[0]*v2[1] - v1[1]*v2[0]

            if abs(cross) > tol:
                simplified.append(b)

        simplified.append(traj[-1])  # Always keep the last point
        return [tuple(pt) for pt in simplified]

    def pose_callback(self, odometry_msg):
        if self.initialized_traj is False:
            return

        x = odometry_msg.pose.pose.position.x
        y = odometry_msg.pose.pose.position.y
        quaternion = [odometry_msg.pose.pose.orientation.x,
                    odometry_msg.pose.pose.orientation.y,
                    odometry_msg.pose.pose.orientation.z,
                    odometry_msg.pose.pose.orientation.w]
        _, _, theta = R.from_quat(quaternion).as_euler('xyz', degrees=False)

        points = self.points
        car = np.array([x, y])

        if self.reverse:
            self.get_logger().info(f'Sending reverse drive command')
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = -1.0
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)
            return

        # Check if at goal
        if np.linalg.norm(points[-1] - car) < 1.0: #0.25
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)
            return

        target_point, _ = self.circle_segment_intersections(points, self.lookahead, car)

        if np.linalg.norm(points[-1] - car) < 2 * self.lookahead:
            target_point = points[-1]
        elif target_point is None:
            target_point = points[np.argmin(np.linalg.norm(points - car, axis=1))]

        steering_angle = self.compute_steering_angle(x, y, target_point[0], target_point[1], theta)

        if self.visualize_intersect is True:
            self.publish_intersection(target_point)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)

    def circle_segment_intersections(self, points, radius, center):
        best_intersection = None
        best_index = -1
        best_t = -1

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            d = p2 - p1
            f = p1 - center

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - radius**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            for sign in [-1, 1]:
                t = (-b + sign * sqrt_disc) / (2 * a)
                if 0 <= t <= 1:
                    if i > best_index or (i == best_index and t > best_t):
                        best_intersection = p1 + t * d
                        best_index = i
                        best_t = t

        return best_intersection, best_index  # will be None if no intersection

    def compute_steering_angle(self, x_robot, y_robot, x_target, y_target, theta):
        """
        Compute the steering angle for pure pursuit.
        """
        dx = x_target - x_robot
        dy = y_target - y_robot
        angle_to_wp = np.arctan2(dy, dx)

        angle = angle_to_wp - theta
        alpha = np.arctan2(np.sin(angle), np.cos(angle))
        steering_angle = np.arctan2(2.0 * self.wheelbase_length * np.sin(alpha),
                                self.lookahead)

        return steering_angle

    def publish_intersection(self, intersection):
        intersect_msg = Marker()
        intersect_msg.header.frame_id = "/map"
        intersect_msg.header.stamp = self.get_clock().now().to_msg()
        intersect_msg.pose.position.x = intersection[0]
        intersect_msg.pose.position.y = intersection[1]
        intersect_msg.pose.position.z = 0.0
        intersect_msg.pose.orientation.x = 0.0
        intersect_msg.pose.orientation.y = 0.0
        intersect_msg.pose.orientation.z = 0.0
        intersect_msg.pose.orientation.w = 0.0
        intersect_msg.scale.x = 1.0
        intersect_msg.scale.y = 1.0
        intersect_msg.scale.z = 1.0

        intersect_msg.type = 2
        intersect_msg.color.r = 0.0
        intersect_msg.color.g = 0.0
        intersect_msg.color.b = 1.0
        intersect_msg.color.a = 1.0

        self.intersect_pub.publish(intersect_msg)

    def reverse_cb(self, msg):
        self.reverse = msg.data

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

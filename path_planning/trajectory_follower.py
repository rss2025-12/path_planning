import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from .utils import LineTrajectory
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion
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
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.intersect_pub = self.create_publisher(Marker,
                                                   "/intersection",
                                                   1)

        self.speed = 5.0  # Remeber to change later
        self.lookahead = 1.0  # 2.25 * self.speed**2
        self.wheelbase_length = 0.1

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.initialized_traj = False
        self.progress_index = 0

        self.visualize_intersect = True

        self.get_logger().info("Path follower initialized")

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

        points = np.array(self.trajectory.points)
        car = np.array([x, y])

        # Check if at goal
        if np.linalg.norm(points[-1] - car) < 0.2:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)
            self.get_logger().info("At goal pose")
            return

        target_point, target_index = self.circle_segment_intersections(points, self.lookahead, car)

        if np.linalg.norm(points[-1] - car) < 2 * self.lookahead:
            target_point = points[-1]
        elif target_point is None:
            target_point = points[np.argmin(np.sum((points - car)**2, axis=1))]

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

    def compute_steering_angle(self, x_robot, y_robot, x_target, y_target, theta):
        """
        Compute the steering angle for pure pursuit.
        """
        desired_heading = np.arctan2(y_target - y_robot, x_target - x_robot)
        steering_angle = desired_heading - theta
        steering_angle = np.arctan2(np.sin(steering_angle), np.cos(steering_angle))
        return steering_angle

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory with {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

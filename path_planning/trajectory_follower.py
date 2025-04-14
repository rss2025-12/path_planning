import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry

from .utils import LineTrajectory
from scipy.spatial.transform import Rotation as R
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

        self.speed = 5.0  # PLEASE GOD REMEMBER TO CHANGE THIS LATER
        self.lookahead = 2.25*self.speed**2
        self.wheelbase_length = 0.1

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.initialized_traj = False

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
        start_to_end = points[1:] - points[:-1]
        start_to_point = np.array([x, y]) - points[:-1]

        dot_products = np.sum(start_to_end * start_to_point, axis=1)
        segment_lengths_squared = np.sum(start_to_end**2, axis=1)
        projections = np.clip(dot_products / segment_lengths_squared, 0, 1)

        closest_points = points[:-1] + projections[:, np.newaxis] * start_to_end
        distances_squared = np.sum((closest_points - np.array([x, y]))**2, axis=1)
        closest_segment_idx = np.argmin(distances_squared)

        if closest_segment_idx < len(self.trajectory.points) - 1: # If point is along path
            target_point = None
            search_idx = closest_segment_idx
            while search_idx < len(self.trajectory.points) - 1:
                closest_wp = self.trajectory.points[search_idx]
                next_wp = self.trajectory.points[search_idx + 1]
                intersect_points = self.circle_segment_intersect([x, y], closest_wp, next_wp)

                if intersect_points:
                    target_point = intersect_points[0]  # Pick the one furthest along the segment
                    break
                search_idx += 1

            # No intersection found, target next waypoint
            if target_point is None:
                target_point = self.trajectory.points[min(closest_segment_idx + 1, len(self.trajectory.points) - 1)]
        else: # If point is last one in path
            target_point = self.trajectory.points[closest_segment_idx]

        steering_angle = self.compute_steering_angle(x, y, target_point[0], target_point[1], theta)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'  # Frame ID
        drive_msg.drive.speed = self.speed  # Linear velocity in m/s
        drive_msg.drive.steering_angle = steering_angle  # Steering angle in radians

        self.drive_pub.publish(drive_msg)

    def circle_segment_intersect(self, circle_center, point1, point2):
        """
        Calculates the intersection point(s) of a circle with a given radius (self.lookahead),
        centered at 'circle_center', and the line segment defined by 'point1' and 'point2'.

        All points are given in [x, y] format.

        Returns:
            A list of intersection point(s) within the segment, sorted by proximity to point2
            (furthest along the segment), or None if no valid intersection exists.
        """
        cx, cy = circle_center
        x1, y1 = point1
        x2, y2 = point2

        dx = x2 - x1
        dy = y2 - y1

        fx = x1 - cx
        fy = y1 - cy

        a = dx**2 + dy**2
        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - self.lookahead**2

        if abs(a) < 1e-8:
            return []

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return []  # No intersection

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                intersections.append([ix, iy])

        intersections.sort(key=lambda p: np.linalg.norm(np.subtract(p, point2)))
        return intersections

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

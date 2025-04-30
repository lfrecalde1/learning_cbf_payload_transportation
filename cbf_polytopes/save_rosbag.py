import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from quadrotor_msgs.msg import TRPYCommand, PositionCommand, MotorSpeed
from scipy.io import savemat
from datetime import datetime
import numpy as np

from cbf_polytopes import cost_translation_casadi, cost_quaternion_casadi

cost_quaternion = cost_quaternion_casadi()
cost_translation = cost_translation_casadi()


class OdomPointSynchronizer(Node): 
    def __init__(self): 
        super().__init__('odom_point_synchronizer') 

        self.last_message_time = self.get_clock().now()
        self.odom_data = []
        self.control_F_data = []
        self.control_M_data = []
        self.position_cmd_data = []
        self.translation_cost_data = []
        self.orientation_cost_data = []
        self.timestamps = []

        self.odom_sub = Subscriber(self, Odometry, '/eagle4/odom')
        self.control_sub = Subscriber(self, TRPYCommand, '/eagle4/trpy_cmd')
        self.position_cmd_sub = Subscriber(self, PositionCommand, '/eagle4/position_cmd')

        self.ts = ApproximateTimeSynchronizer([self.odom_sub, self.control_sub, self.position_cmd_sub], queue_size=20, slop=0.5)
        self.ts.registerCallback(self.sync_callback)

        self.timer = self.create_timer(2.0, self.check_timeout)

    def sync_callback(self, odom_msg, control_msg, desired_msg):
        self.last_message_time = self.get_clock().now()

        if not desired_msg.points:
            self.get_logger().warn("Received PositionCommand with no points.")
            return

        pose = np.array([
            odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z,
            odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z,
            odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z,
            odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z
        ])

        control_F = np.array([control_msg.thrust])
        control_M = np.array([
            control_msg.angular_velocity.x,
            control_msg.angular_velocity.y,
            control_msg.angular_velocity.z
        ])

        desired = np.array([
            desired_msg.position.x, desired_msg.position.y, desired_msg.position.z,
            desired_msg.velocity.x, desired_msg.velocity.y, desired_msg.velocity.z,
            desired_msg.points[0].quaternion.w, desired_msg.points[0].quaternion.x, desired_msg.points[0].quaternion.y, desired_msg.points[0].quaternion.z,
            desired_msg.points[0].angular_velocity.x, desired_msg.points[0].angular_velocity.y, desired_msg.points[0].angular_velocity.z
        ])

        orientation_cost = cost_quaternion(desired[6:10], pose[6:10])
        translation_cost = cost_translation(desired[0:3], pose[0:3])

        self.odom_data.append(pose)
        self.control_F_data.append(control_F)
        self.control_M_data.append(control_M)
        self.position_cmd_data.append(desired)
        self.translation_cost_data.append(float(translation_cost[0, 0]))
        self.orientation_cost_data.append(float(orientation_cost[0, 0]))
        self.timestamps.append(odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9)

        self.get_logger().info(f"Synchronized Messages:\n"
                               f"Odom Timestamp: {odom_msg.header.stamp.sec}.{odom_msg.header.stamp.nanosec}\n"
                               f"Control Timestamp: {control_msg.header.stamp.sec}.{control_msg.header.stamp.nanosec}\n"
                               f"Cmd Timestamp: {desired_msg.header.stamp.sec}.{desired_msg.header.stamp.nanosec}\n")

    def check_timeout(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.last_message_time).nanoseconds / 1e9

        if elapsed_time > 15.0:
            self.get_logger().info("No new messages received. Exiting and saving data.")
            self.save_data_to_mat()
            rclpy.shutdown()

    def save_data_to_mat(self):
        if not self.timestamps:
            self.get_logger().info("No data to save.")
            return

        data_dict = {
            "x": np.array(self.odom_data).T,
            "F": np.array(self.control_F_data).T,
            "M": np.array(self.control_M_data).T,
            "xref": np.array(self.position_cmd_data).T,
            "translation_cost": np.array(self.translation_cost_data).T,
            "orientation_cost": np.array(self.orientation_cost_data).T,
            "timestamps": self.timestamps
        }

        filename = f"Dq_lissajous{4_5}.mat"
        savemat(filename, data_dict)
        self.get_logger().info(f"Data saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = OdomPointSynchronizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user. Exiting...")
        node.save_data_to_mat()
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
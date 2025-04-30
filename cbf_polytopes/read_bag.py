import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TRPYCommand, PositionCommand
from scipy.io import savemat
from datetime import datetime
import numpy as np
import os 
from cbf_polytopes import cost_translation_casadi, cost_quaternion_casadi

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

cost_quaternion = cost_quaternion_casadi()
cost_translation = cost_translation_casadi()


class OdomPointSynchronizer(Node):
    def __init__(self, rosbag_path):
        super().__init__('odom_point_synchronizer')

        self.last_message_time = self.get_clock().now()

        self.odom_data = []
        self.control_F_data = []
        self.control_M_data = []
        self.position_cmd_data = []
        self.translation_cost_data = []
        self.orientation_cost_data = []
        self.timestamps = []

        self.rosbag_path = rosbag_path

        self.start_reading_rosbag()

    def start_reading_rosbag(self):
        storage_options = StorageOptions(uri=self.rosbag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}

        # Initialize caches
        self.cached_odom = None
        self.cached_control = None
        self.cached_desired = None

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == '/eagle4/odom':
                self.cached_odom = deserialize_message(data, Odometry)
            elif topic == '/eagle4/trpy_cmd':
                self.cached_control = deserialize_message(data, TRPYCommand)
            elif topic == '/eagle4/position_cmd':
                self.cached_desired = deserialize_message(data, PositionCommand)
            else:
                continue

            # Check if we have all three cached
            if self.cached_odom and self.cached_control and self.cached_desired:
                self.sync_callback(self.cached_odom, self.cached_control, self.cached_desired)
                # Reset caches
                self.cached_odom = None
                self.cached_control = None
                self.cached_desired = None

        # After finishing reading
        self.save_data_to_mat()
        self.get_logger().info("Finished reading bag and saving data.")
        self.destroy_node()
        rclpy.shutdown()

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
        folder_name = os.path.basename(os.path.normpath(self.rosbag_path))
        filename = f"{folder_name}.mat"
        savemat(filename, data_dict)
        self.get_logger().info(f"Data saved to {filename}")


def main(args=None):
    rclpy.init(args=args)

    # Specify your rosbag path
    #rosbag_path = './april_27_baseline_nmpc_liss_3_0_z'   # <<--- EDIT this with your actual folder path!
    rosbag_path = './april_27_baseline_nmpc_circle_slow'   # <<--- EDIT this with your actual folder path!

    node = OdomPointSynchronizer(rosbag_path)

    # No spin needed — reading finishes automatically


if __name__ == '__main__':
    main()
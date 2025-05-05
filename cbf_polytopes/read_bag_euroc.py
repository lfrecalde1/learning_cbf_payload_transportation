import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
import numpy as np
import os
import cv2
import csv

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

class EurocFormatterFullRate(Node):
    def __init__(self, rosbag_path):
        super().__init__('euroc_formatter_full_rate')

        # Data containers
        self.all_odom_data = []
        self.all_imu_data = []
        self.all_image_data = []

        self.rosbag_path = rosbag_path

        # Output folders
        self.output_folder = os.path.join(self.rosbag_path, "euroc_dataset")

        self.cam_folder = os.path.join(self.output_folder, "cam0")
        self.imu_folder = os.path.join(self.output_folder, "imu0")
        self.gt_folder = os.path.join(self.output_folder, "state_groundtruth_estimate0")

        self.cam_data_folder = os.path.join(self.cam_folder, "data")

        os.makedirs(self.cam_data_folder, exist_ok=True)
        os.makedirs(self.imu_folder, exist_ok=True)
        os.makedirs(self.gt_folder, exist_ok=True)

        self.start_reading_rosbag()

    def start_reading_rosbag(self):
        storage_options = StorageOptions(uri=self.rosbag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}
        self.get_logger().info(f"Found topics: {list(type_map.keys())}")

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == '/odom':
                odom_msg = deserialize_message(data, Odometry)
                self.handle_odom(odom_msg)

            elif topic == '/imu':
                imu_msg = deserialize_message(data, Imu)
                self.handle_imu(imu_msg)

            elif topic == '/rgb_image':
                img_msg = deserialize_message(data, Image)
                self.handle_image(img_msg)

        # After reading all data, save to CSVs
        self.save_imu_csv()
        self.save_images_and_csv()
        self.save_odom_csv()

        self.get_logger().info("Finished saving all data in EuRoC format.")
        self.destroy_node()
        rclpy.shutdown()

    def handle_odom(self, odom_msg):
        t = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9

        pose = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
            odom_msg.orientation.w,
            odom_msg.orientation.x,
            odom_msg.orientation.y,
            odom_msg.orientation.z,
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z
        ]

        timestamp_ns = int(t * 1e9)
        self.all_odom_data.append((timestamp_ns, pose))

    def handle_imu(self, imu_msg):
        t = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9

        imu = [
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ]

        timestamp_ns = int(t * 1e9)
        self.all_imu_data.append((timestamp_ns, imu))

    def handle_image(self, img_msg):
        t = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        timestamp_ns = int(t * 1e9)

        img_np = np.frombuffer(img_msg.data, dtype=np.uint8)

        if img_msg.encoding in ['rgb8', 'bgr8']:
            img_np = img_np.reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding in ['mono8']:
            img_np = img_np.reshape((img_msg.height, img_msg.width))
        else:
            self.get_logger().warn(f"Unsupported image encoding: {img_msg.encoding}")
            return

        self.all_image_data.append((timestamp_ns, img_np))

    def save_imu_csv(self):
        imu_csv_path = os.path.join(self.imu_folder, "data.csv")

        with open(imu_csv_path, mode='w', newline='') as imu_file:
            writer = csv.writer(imu_file)
            writer.writerow(["#timestamp [ns]", "w_x [rad/s]", "w_y [rad/s]", "w_z [rad/s]",
                             "a_x [m/s^2]", "a_y [m/s^2]", "a_z [m/s^2]"])

            for timestamp_ns, imu in self.all_imu_data:
                writer.writerow([timestamp_ns, *imu])

    def save_images_and_csv(self):
        cam_csv_path = os.path.join(self.cam_folder, "data.csv")

        with open(cam_csv_path, mode='w', newline='') as cam_file:
            writer = csv.writer(cam_file)
            writer.writerow(["#timestamp [ns]", "filename"])

            for timestamp_ns, img_np in self.all_image_data:
                filename = f"{timestamp_ns}.jpg"
                img_path = os.path.join(self.cam_data_folder, filename)

                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_to_save = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_to_save = img_np

                cv2.imwrite(img_path, img_to_save)
                writer.writerow([timestamp_ns, filename])

    def save_odom_csv(self):
        gt_csv_path = os.path.join(self.gt_folder, "data.csv")

        with open(gt_csv_path, mode='w', newline='') as gt_file:
            writer = csv.writer(gt_file)
            writer.writerow(["#timestamp [ns]",
                             "p_x [m]", "p_y [m]", "p_z [m]",
                             "q_w", "q_x", "q_y", "q_z",
                             "v_x [m/s]", "v_y [m/s]", "v_z [m/s]"])

            for timestamp_ns, pose in self.all_odom_data:
                writer.writerow([timestamp_ns, *pose])


def main(args=None):
    rclpy.init(args=args)

    rosbag_path = './liss'  # <<< Replace with your bag folder
    node = EurocFormatterFullRate(rosbag_path)

if __name__ == '__main__':
    main()
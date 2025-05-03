import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from scipy.io import savemat
import numpy as np
import os 
import cv2

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

class OdomImuImageSynchronizer(Node):
    def __init__(self, rosbag_path):
        super().__init__('odom_imu_image_synchronizer')

        # All data storage (to be aligned later)
        self.all_odom_data = []
        self.all_odom_timestamps = []

        self.all_imu_data = []
        self.all_imu_timestamps = []

        self.image_timestamps = []
        self.image_frames = []

        # Final synchronized data
        self.synced_odom_data = []
        self.synced_odom_timestamps = []

        self.synced_imu_data = []
        self.synced_imu_timestamps = []

        self.synced_image_timestamps = []

        self.synced_imu_window = []  # 10 past IMU readings per image

        self.rosbag_path = rosbag_path

        self.image_output_folder = os.path.join(self.rosbag_path, "saved_images")
        os.makedirs(self.image_output_folder, exist_ok=True)

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

        # Match data once all messages have been read
        self.match_data()
        self.save_data_to_mat()

        self.get_logger().info("Finished reading bag and saving data.")
        self.destroy_node()
        rclpy.shutdown()

    def handle_odom(self, odom_msg):
        pose = np.array([
            odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z,
            odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z,
            odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z,
            odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z
        ])

        t = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9

        self.all_odom_data.append(pose)
        self.all_odom_timestamps.append(t)

    def handle_imu(self, imu_msg):
        imu = np.array([
            imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z
        ])

        t = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9

        self.all_imu_data.append(imu)
        self.all_imu_timestamps.append(t)

    def handle_image(self, img_msg):
        img_time = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        self.image_timestamps.append(img_time)

        img_np = np.frombuffer(img_msg.data, dtype=np.uint8)

        if img_msg.encoding in ['rgb8', 'bgr8']:
            img_np = img_np.reshape((img_msg.height, img_msg.width, 3))
        elif img_msg.encoding in ['mono8']:
            img_np = img_np.reshape((img_msg.height, img_msg.width))
        else:
            self.get_logger().warn(f"Unsupported image encoding: {img_msg.encoding}")
            return

        self.image_frames.append(img_np)

        img_filename = os.path.join(self.image_output_folder, f"image_{len(self.image_frames):05d}.png")

        if img_msg.encoding == 'rgb8':
            img_to_save = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img_np

        cv2.imwrite(img_filename, img_to_save)

    def match_data(self):
        odom_times = np.array(self.all_odom_timestamps)
        imu_times = np.array(self.all_imu_timestamps)

        for idx, img_time in enumerate(self.image_timestamps):

            # --- Find closest odometry ---
            odom_diffs = np.abs(odom_times - img_time)
            odom_idx = int(np.argmin(odom_diffs))
            matched_odom = self.all_odom_data[odom_idx]
            matched_odom_time = self.all_odom_timestamps[odom_idx]

            # --- Find closest imu ---
            imu_diffs = np.abs(imu_times - img_time)
            imu_idx = int(np.argmin(imu_diffs))
            matched_imu = self.all_imu_data[imu_idx]
            matched_imu_time = self.all_imu_timestamps[imu_idx]

            # --- Collect past 10 IMU measurements ---
            imu_window = []
            count = 0
            i = imu_idx
            while count < 10 and i >= 0:
                imu_window.insert(0, self.all_imu_data[i])
                count += 1
                i -= 1

            # If less than 10 IMU available (start of bag), pad by repeating earliest value
            while len(imu_window) < 10:
                imu_window.insert(0, imu_window[0])

            imu_window_np = np.stack(imu_window, axis=1)  # Shape (6,10)

            # --- Save synced data ---
            self.synced_odom_data.append(matched_odom)
            self.synced_odom_timestamps.append(matched_odom_time)

            self.synced_imu_data.append(matched_imu)
            self.synced_imu_timestamps.append(matched_imu_time)

            self.synced_image_timestamps.append(img_time)

            self.synced_imu_window.append(imu_window_np)  # (6,10)

            self.get_logger().info(
                f"Image {idx} time {img_time:.3f} matched to ODOM {matched_odom_time:.3f} and IMU {matched_imu_time:.3f}"
            )

    def save_data_to_mat(self):
        if not self.synced_image_timestamps:
            self.get_logger().info("No data to save.")
            return

        imu_window_array = np.stack(self.synced_imu_window, axis=2)  # (6,10,N)

        data_dict = {
            "x": np.array(self.synced_odom_data).T,    # (12, N)
            "imu": np.array(self.synced_imu_data).T,   # (6, N)
            "imu_window": imu_window_array,            # (6,10,N)
            "odom_timestamps": self.synced_odom_timestamps,
            "imu_timestamps": self.synced_imu_timestamps,
            "image_timestamps": self.synced_image_timestamps
        }

        folder_name = os.path.basename(os.path.normpath(self.rosbag_path))
        filename = f"{folder_name}.mat"
        savemat(filename, data_dict)
        self.get_logger().info(f"Data saved to {filename}")


def main(args=None):
    rclpy.init(args=args)

    rosbag_path = './trajectories'   # <<< Replace with your bag folder
    node = OdomImuImageSynchronizer(rosbag_path)

if __name__ == '__main__':
    main()
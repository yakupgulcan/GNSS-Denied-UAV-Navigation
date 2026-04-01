#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import csv
import time
import os
from datetime import datetime
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class BenchmarkLogger(Node):
    def __init__(self):
        super().__init__('benchmark_logger')
        
        # --- Settings ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f'benchmark_data_{timestamp}.csv'
        self.file_path = os.path.join(os.getcwd(), self.filename)

        # Initialize CSV file
        self.csv_file = open(self.file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Columns: timestamp, ground-truth X/Y, estimated X/Y
        self.csv_writer.writerow(['timestamp', 'true_x', 'true_y', 'est_x', 'est_y'])

        self.get_logger().info(f"Logging started: {self.filename}")
        self.get_logger().info("Waiting for topics: /odometry and /visual_pose_enu")

        # --- State variables ---
        self.start_offset_x = 0.0
        self.start_offset_y = 0.0
        self.is_initialized = False  # True once the first odometry message is received

        self.true_x = 0.0
        self.true_y = 0.0
        self.est_x = 0.0
        self.est_y = 0.0

        # --- Subscriptions ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # 1. Ground-truth position (Gazebo odometry)
        self.create_subscription(Odometry, '/odometry', self.odom_cb, qos_profile)

        # 2. Estimated position (visual localization algorithm)
        self.create_subscription(PoseStamped, '/visual_pose_enu', self.visual_cb, 10)

        # --- Logging timer (10 Hz) ---
        self.timer = self.create_timer(0.1, self.log_data)

    def odom_cb(self, msg):
        # Extract position from odometry message
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y

        # On first message, record the origin offset to normalize to (0, 0)
        if not self.is_initialized:
            self.start_offset_x = raw_x
            self.start_offset_y = raw_y
            self.is_initialized = True
            self.get_logger().info(
                f"Origin set (Gazebo): X={raw_x:.2f}, Y={raw_y:.2f}")

        # Apply offset so trajectory starts at (0, 0)
        self.true_x = raw_x - self.start_offset_x
        self.true_y = raw_y - self.start_offset_y

    def visual_cb(self, msg):
        # Visual estimate already starts at (0, 0), take it directly
        self.est_x = msg.pose.position.x
        self.est_y = msg.pose.position.y

    def log_data(self):
        # Do not log until the origin has been established
        if not self.is_initialized:
            return

        t = time.time()
        self.csv_writer.writerow([t, self.true_x, self.true_y, self.est_x, self.est_y])
        self.csv_file.flush() 

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
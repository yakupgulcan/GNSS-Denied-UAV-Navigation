#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import csv
import time
import os
from datetime import datetime

class AdvancedLogger(Node):
    def __init__(self):
        super().__init__('advanced_benchmark_logger')
        
        # File setup
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f'perf_metrics_{ts}.csv'
        self.csv_file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        # Columns: timestamp, algo, proc_time_ms, status, estimated XY, ground truth XY, error
        self.writer.writerow(['timestamp', 'algo_name', 'proc_time_ms', 'status', 'est_x', 'est_y', 'true_x', 'true_y', 'error_m'])

        self.get_logger().info(f"Logging started: {self.filename}")

        # Data buffers
        self.current_true_x = 0.0
        self.current_true_y = 0.0
        self.start_offset_x = 0.0
        self.start_offset_y = 0.0
        self.is_initialized = False

        # Subscriptions
        self.create_subscription(Odometry, '/odometry', self.odom_cb, 10)
        self.create_subscription(String, '/visual_perf_metrics', self.metrics_cb, 10)

    def odom_cb(self, msg):
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        
        if not self.is_initialized:
            self.start_offset_x = raw_x
            self.start_offset_y = raw_y
            self.is_initialized = True
        
        self.current_true_x = raw_x - self.start_offset_x
        self.current_true_y = raw_y - self.start_offset_y

    def metrics_cb(self, msg):
        if not self.is_initialized: return

        # Expected format: "algo_name,status,duration_ms,est_x,est_y"
        try:
            parts = msg.data.split(',')
            algo_name = parts[0]
            status = parts[1]
            duration = float(parts[2])
            est_x = float(parts[3])
            est_y = float(parts[4])

            # Compute error (only meaningful when a match was found)
            error = 0.0
            if "MATCH" in status:  # Covers both "MATCH" and "MATCH_INIT"
                error = ((est_x - self.current_true_x)**2 + (est_y - self.current_true_y)**2)**0.5

            t = time.time()
            self.writer.writerow([t, algo_name, duration, status, est_x, est_y,
                                   self.current_true_x, self.current_true_y, error])
            self.csv_file.flush()

        except Exception as e:
            self.get_logger().error(f"Parse error: {e}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
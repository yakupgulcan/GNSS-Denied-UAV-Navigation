#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import os
import time
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

"""
Description:
---
This node receives images from the drone camera, synchronizes them with GPS
and heading data from MAVROS, and saves them to disk.

Functionality:
1. Subscribes to /camera/image for the camera stream.
2. Subscribes to /mavros/global_position/global for GPS data.
3. Subscribes to /mavros/global_position/compass_hdg for heading (0-360°).
4. Writes frames to disk at a configurable interval (default ~0.67 FPS).
5. Creates a JSON metadata record for each saved frame.
---
"""


class ImageSaver(Node):
    """ROS 2 node that synchronously saves camera frames and telemetry data."""

    def __init__(self):
        super().__init__("image_saver")

        # --- 1. File system setup ---
        # Create a timestamped directory to avoid overwriting previous runs
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.expanduser(f"~/frames_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.json_path = os.path.join(self.output_dir, "frames_details.json")
        self.frames_data = []  # In-memory list of frame metadata

        # --- 2. Utilities ---
        self.bridge = CvBridge()
        self.frame_count = 0

        # --- 3. Telemetry state ---
        self.current_gps = {"lat": None, "lon": None, "alt": None}
        self.current_heading = None

        # --- 4. QoS settings ---
        # MAVROS publishes with BEST_EFFORT; use the same to avoid subscription mismatches.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- 5. Subscriptions ---
        self.create_subscription(Image, "/camera/image", self.image_callback, 10)
        self.create_subscription(NavSatFix, "/mavros/global_position/global",
                                 self.gps_callback, qos_profile=qos)
        self.create_subscription(Float64, "/mavros/global_position/compass_hdg",
                                 self.heading_callback, qos_profile=qos)

        # --- 6. Save timer (~0.67 FPS) ---
        self.timer = self.create_timer(1.5, self.save_frame)
        self.latest_image = None

        self.get_logger().info(f"Image saver started. Output: {self.output_dir}")

    def gps_callback(self, msg: NavSatFix):
        """Update GPS coordinates (lat, lon, alt)."""
        self.current_gps["lat"] = msg.latitude
        self.current_gps["lon"] = msg.longitude
        self.current_gps["alt"] = msg.altitude

    def heading_callback(self, msg: Float64):
        """Update compass heading (0–360°)."""
        self.current_heading = msg.data

    def image_callback(self, msg: Image):
        """Convert incoming ROS Image message to OpenCV BGR format."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")

    def save_frame(self):
        """Timer callback: save the latest image and current telemetry to disk."""
        if self.latest_image is None:
            return

        # Build filename with timestamp and zero-padded frame index
        filename = f"frame_{self.timestamp}_{self.frame_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, self.latest_image)

        # Build metadata record
        frame_info = {
            "filename": filename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gps": self.current_gps.copy(),  # Copy dict to avoid reference aliasing
            "heading": self.current_heading
        }
        self.frames_data.append(frame_info)

        # Overwrite JSON with the full list to prevent data loss on crash
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.frames_data, f, indent=4)
        except Exception as e:
            self.get_logger().error(f"JSON write error: {e}")

        self.frame_count += 1

        # Log every 10 frames to avoid flooding the terminal
        if self.frame_count % 10 == 0:
            hdg_str = f"{self.current_heading:.2f}" if self.current_heading else "N/A"
            self.get_logger().info(
                f"Saved: {self.frame_count} frames | File: {filename} | Hdg: {hdg_str}")

    def destroy_node(self):
        """Flush metadata JSON on shutdown."""
        if self.frames_data:
            with open(self.json_path, "w") as f:
                json.dump(self.frames_data, f, indent=4)
        self.get_logger().info(
            f"Shutting down. Total frames saved: {self.frame_count}")
        self.get_logger().info(f"JSON metadata: {self.json_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted (Ctrl+C). Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
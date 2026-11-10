#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float64, Bool
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
Only saves frames once the drone has reached flight altitude to prevent
corrupting the feature database with ground/takeoff frames.
---
"""


class ImageSaver(Node):
    """ROS 2 node that synchronously saves camera frames and telemetry data."""

    def __init__(self):
        super().__init__("image_saver")

        # --- Parameters ---
        self.declare_parameter("min_record_alt", 25.0)
        self.declare_parameter("save_interval", 1.5)
        self.min_record_alt = float(self.get_parameter("min_record_alt").value)
        save_interval = float(self.get_parameter("save_interval").value)

        # --- 1. File system setup ---
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.expanduser(f"~/frames_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.json_path = os.path.join(self.output_dir, "frames_details.json")
        self.frames_data = []

        # --- 2. Utilities ---
        self.bridge = CvBridge()
        self.frame_count = 0

        # --- 3. Telemetry & Recording state ---
        self.current_gps = {"lat": None, "lon": None, "alt": None}
        self.current_heading = None
        self.current_rel_alt = 0.0
        self.recording_active = True

        # --- 4. QoS settings ---
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
        self.create_subscription(Float64, "/mavros/global_position/rel_alt",
                                 self.alt_callback, qos_profile=qos)
        self.create_subscription(Bool, "/dataset/record",
                                 self.record_control_callback, 10)

        # --- 6. Save timer ---
        self.timer = self.create_timer(save_interval, self.save_frame)
        self.latest_image = None

        self.get_logger().info(
            f"Image saver started. Output: {self.output_dir} | Min alt: {self.min_record_alt:.1f}m | Interval: {save_interval:.2f}s"
        )

    def record_control_callback(self, msg: Bool):
        """Enable or disable image recording based on mission state."""
        if not msg.data and self.recording_active:
            self.recording_active = False
            if self.timer is not None:
                self.timer.cancel()
            self.flush_metadata()
            self.get_logger().info("🛑 Survey scan complete! Image recording stopped by mission controller.")
            self.get_logger().info(f"✅ Total frames saved: {self.frame_count}")
            self.get_logger().info(f"📁 Output directory: {self.output_dir}")
            self.get_logger().info(f"📄 Metadata written: {self.json_path}")
        elif msg.data and not self.recording_active:
            self.recording_active = True
            self.get_logger().info("▶️ Image recording resumed.")

    def gps_callback(self, msg: NavSatFix):
        """Update GPS coordinates (lat, lon, alt)."""
        self.current_gps["lat"] = msg.latitude
        self.current_gps["lon"] = msg.longitude
        self.current_gps["alt"] = msg.altitude

    def heading_callback(self, msg: Float64):
        """Update compass heading (0–360°)."""
        self.current_heading = msg.data

    def alt_callback(self, msg: Float64):
        """Update relative altitude above takeoff."""
        self.current_rel_alt = msg.data

    def image_callback(self, msg: Image):
        """Convert incoming ROS Image message to OpenCV BGR format."""
        if not self.recording_active:
            return
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")

    def flush_metadata(self):
        """Write current metadata JSON to disk."""
        if self.frames_data:
            try:
                with open(self.json_path, "w") as f:
                    json.dump(self.frames_data, f, indent=4)
            except Exception as e:
                self.get_logger().error(f"JSON write error: {e}")

    def save_frame(self):
        """Timer callback: save the latest image and current telemetry to disk."""
        if not self.recording_active or self.latest_image is None:
            return

        # Do not save frames on the ground or during initial takeoff climb
        if self.current_rel_alt < self.min_record_alt:
            return

        # Ensure valid GPS fix before recording
        if self.current_gps["lat"] is None or self.current_gps["lon"] is None:
            return

        # Build filename with timestamp and zero-padded frame index
        filename = f"frame_{self.timestamp}_{self.frame_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, self.latest_image)

        # Build metadata record
        frame_info = {
            "filename": filename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gps": self.current_gps.copy(),
            "heading": self.current_heading,
            "rel_alt": self.current_rel_alt
        }
        self.frames_data.append(frame_info)

        # Overwrite JSON with the full list to prevent data loss on crash
        self.flush_metadata()

        self.frame_count += 1

        # Log every 10 frames to avoid flooding the terminal
        if self.frame_count % 10 == 0:
            hdg_str = f"{self.current_heading:.2f}" if self.current_heading is not None else "N/A"
            self.get_logger().info(
                f"Saved: {self.frame_count} frames | File: {filename} | Alt: {self.current_rel_alt:.1f}m | Hdg: {hdg_str}")

    def destroy_node(self):
        """Flush metadata JSON on shutdown."""
        self.flush_metadata()
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

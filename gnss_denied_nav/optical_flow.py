#!/usr/bin/env python3
"""
Optical Flow Velocity Node

Estimates horizontal ground velocity from the nadir camera: features are
tracked frame to frame (Lucas-Kanade), a RANSAC similarity transform gives
the camera's image-plane motion (so yaw rotation does not leak into
velocity), and the motion is scaled to meters by height / focal length.

Time steps come from the image header stamps, so velocities are in meters
per simulated second even when Gazebo runs slower than real time.

Outputs (stamped with the image time):
    /visual_velocity_enu   (east, north) m/s
    /visual_velocity_body  (forward, left) m/s, ROS FLU convention
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from gnss_denied_nav.localization.camera import CameraModel


class FlowOdometry:
    """Frame-to-frame camera velocity from a nadir image stream (ROS-free)."""

    def __init__(self, camera, max_corners=200, min_tracks=60, ransac_px=1.0):
        self.camera = camera
        self.min_tracks = min_tracks
        self.ransac_px = ransac_px
        self.feature_params = dict(maxCorners=max_corners, qualityLevel=0.05,
                                   minDistance=12, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_stamp = None

    def _detect(self, gray, existing=None):
        mask = None
        if existing is not None and len(existing):
            mask = np.full(gray.shape, 255, np.uint8)
            for x, y in existing.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if new is None:
            return existing
        return new if existing is None or not len(existing) else np.vstack([existing, new])

    def update(self, gray, stamp, height, heading_deg):
        """Feed one frame. Returns (v_east, v_north, v_fwd, v_right) in m/s, or None."""
        result = None
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 8:
            dt = stamp - self.prev_stamp
            if dt <= 0.0:
                return None
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None,
                                                 **self.lk_params)
            ok = st.ravel() == 1
            p0, p1 = self.prev_pts[ok], p1[ok]
            tracked = p1
            if len(p0) >= 8:
                A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=self.ransac_px)
                if A is not None and inl is not None and inl.sum() >= 8:
                    tracked = p1[inl.ravel() == 1]
                    result = self._velocity(A, dt, height, heading_deg)
            self.prev_pts = tracked
        if self.prev_pts is None or len(self.prev_pts) < self.min_tracks:
            self.prev_pts = self._detect(gray, self.prev_pts)
        self.prev_gray = gray
        self.prev_stamp = stamp
        return result

    def _velocity(self, A, dt, height, heading_deg):
        # Where the current image center lies in the previous image = camera motion.
        M = np.vstack([A, [0.0, 0.0, 1.0]])
        cx, cy = self.camera.center
        u, v, _ = np.linalg.solve(M, [cx, cy, 1.0])
        gsd = self.camera.gsd(height)
        fwd = -(v - cy) * gsd / dt
        right = (u - cx) * gsd / dt
        h = math.radians(heading_deg)
        east = fwd * math.sin(h) + right * math.cos(h)
        north = fwd * math.cos(h) - right * math.sin(h)
        return east, north, fwd, right


class VisualVelocityNode(Node):

    def __init__(self):
        super().__init__('visual_velocity_estimator')

        # Effective camera FOV: CameraZoomPlugin drives the gimbal camera to
        # 2.0 rad regardless of the SDF value (see localization/camera.py).
        self.declare_parameter('camera_hfov', 2.0)
        self.declare_parameter('publish_rate', 10.0)   # max processed frames per (sim) second
        # Low-pass filter on the ENU velocity (1.0 = no filtering).
        self.declare_parameter('lpf_alpha', 0.5)
        self.declare_parameter('min_altitude', 1.0)

        self.hfov = float(self.get_parameter('camera_hfov').value)
        self.min_period = 1.0 / float(self.get_parameter('publish_rate').value)
        self.alpha = float(self.get_parameter('lpf_alpha').value)
        self.min_altitude = float(self.get_parameter('min_altitude').value)

        self.flow = None
        self.bridge = CvBridge()
        self.altitude = 0.0
        self.heading = None
        self.last_stamp = None
        self.v_east = self.v_north = self.v_fwd = self.v_right = 0.0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/image', self.image_callback, qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg',
                                 self.heading_callback, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_enu', 10)
        self.body_vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_body', 10)
        self.get_logger().info(
            f'Optical flow started (hfov {self.hfov:.2f} rad, alpha {self.alpha}).')

    def alt_cb(self, msg):
        self.altitude = msg.data

    def heading_callback(self, msg):
        self.heading = msg.data

    def image_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and stamp - self.last_stamp < self.min_period:
            return
        self.last_stamp = stamp
        if self.heading is None:
            return
        if self.flow is None:
            self.flow = FlowOdometry(CameraModel(msg.width, msg.height, self.hfov))

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        if self.altitude < self.min_altitude:
            # Too close to the ground for meaningful flow: report zero, restart tracking.
            self.flow.reset()
            self.v_east = self.v_north = self.v_fwd = self.v_right = 0.0
        else:
            vel = self.flow.update(gray, stamp, self.altitude, self.heading)
            if vel is None:
                return
            a = self.alpha
            self.v_east = a * vel[0] + (1 - a) * self.v_east
            self.v_north = a * vel[1] + (1 - a) * self.v_north
            self.v_fwd = a * vel[2] + (1 - a) * self.v_fwd
            self.v_right = a * vel[3] + (1 - a) * self.v_right

        enu = TwistStamped()
        enu.header.stamp = msg.header.stamp
        enu.header.frame_id = 'map'
        enu.twist.linear.x = float(self.v_east)
        enu.twist.linear.y = float(self.v_north)
        self.vel_pub.publish(enu)

        body = TwistStamped()
        body.header.stamp = msg.header.stamp
        body.header.frame_id = 'base_link'
        body.twist.linear.x = float(self.v_fwd)
        body.twist.linear.y = float(-self.v_right)
        self.body_vel_pub.publish(body)


def main(args=None):
    rclpy.init(args=args)
    node = VisualVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

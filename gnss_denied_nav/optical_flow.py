#!/usr/bin/env python3
"""
Optical Flow Velocity Node

Estimates horizontal ground velocity from the nadir camera: features are
tracked frame to frame (Lucas-Kanade), a RANSAC similarity transform gives
the camera's image-plane motion (so yaw rotation does not leak into
velocity), and the motion is scaled to meters by height / focal length.

Time steps come from the image header stamps, so velocities are in meters
per simulated second even when Gazebo runs slower than real time.

Heading (heading_source):
    map      compass + an offset tracked from the absolute camera heading of the
             map fixes (/visual_nav/diagnostics). Default. The ENU velocity may
             be fed to the ArduPilot EKF (GPS_INPUT); rotating it with the EKF's
             own heading would close a loop in which an EKF yaw error makes the
             "measured" velocity agree with it, so the error grows (seen in
             flight: -8 deg). The map heading is independent of the EKF.
    compass  /mavros/global_position/compass_hdg only (EKF heading).

Rotation compensation (rate_compensation):
    A camera rotation shifts the image like a translation would: height x
    angular rate of apparent ground speed. The gimbal removes most of the body
    rotation but not all; in flight the residual flow error was
    0.061 x roll rate x height (lateral) and -0.059 x pitch rate x height
    (forward), and with the flow fed to the EKF it drove a 0.38 Hz roll
    oscillation. Body rates come from the autopilot IMU (/mavros/imu/data,
    FLU); ATTITUDE is requested at attitude_rate_hz. The coefficients are a
    property of the camera mount: ~1 for a body-fixed camera, ~0 for a perfect
    gimbal.

Outputs (stamped with the image time):
    /visual_velocity_enu   (east, north) m/s
    /visual_velocity_body  (forward, left) m/s, ROS FLU convention
"""
import json
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from mavros_msgs.srv import MessageInterval
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float64, String

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
        east, north = body_to_enu(fwd, right, heading_deg)
        return east, north, fwd, right


def body_to_enu(fwd, right, heading_deg):
    """(forward, right) velocity in the camera/body frame -> (east, north)."""
    h = math.radians(heading_deg)
    return fwd * math.sin(h) + right * math.cos(h), fwd * math.cos(h) - right * math.sin(h)


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
        self.declare_parameter('heading_source', 'map')    # 'map' or 'compass'
        self.declare_parameter('heading_offset_gain', 0.05)  # per map fix (~2 s at 10 Hz)
        self.declare_parameter('heading_min_inliers', 30)
        self.declare_parameter('heading_max_jump_deg', 20.0)  # reject fixes this far off the offset
        # Remove the apparent ground motion caused by residual camera rotation.
        self.declare_parameter('rate_compensation', True)
        self.declare_parameter('rate_comp_roll', 0.061)     # lateral error / (roll rate * height)
        self.declare_parameter('rate_comp_pitch', -0.059)   # forward error / (pitch rate * height)
        self.declare_parameter('attitude_rate_hz', 50.0)    # requested ATTITUDE stream rate

        self.hfov = float(self.get_parameter('camera_hfov').value)
        self.min_period = 1.0 / float(self.get_parameter('publish_rate').value)
        self.alpha = float(self.get_parameter('lpf_alpha').value)
        self.min_altitude = float(self.get_parameter('min_altitude').value)
        self.heading_source = str(self.get_parameter('heading_source').value)
        self.offset_gain = float(self.get_parameter('heading_offset_gain').value)
        self.heading_min_inliers = int(self.get_parameter('heading_min_inliers').value)
        self.heading_max_jump = float(self.get_parameter('heading_max_jump_deg').value)
        self.rate_compensation = bool(self.get_parameter('rate_compensation').value)
        self.k_roll = float(self.get_parameter('rate_comp_roll').value)
        self.k_pitch = float(self.get_parameter('rate_comp_pitch').value)
        self.attitude_rate = float(self.get_parameter('attitude_rate_hz').value)
        if self.heading_source not in ('map', 'compass'):
            raise ValueError(f"heading_source must be 'map' or 'compass', not {self.heading_source!r}")

        self.flow = None
        self.bridge = CvBridge()
        self.altitude = 0.0
        self.heading = None           # compass (EKF) heading, deg
        self.heading_offset = None    # map heading - compass, deg
        self.last_offset_log = 0.0
        self.last_stamp = None
        self.gyro = deque(maxlen=200)  # (wall time, roll rate, pitch rate) rad/s, FLU
        self.v_east = self.v_north = self.v_fwd = self.v_right = 0.0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/camera/image', self.image_callback, qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg',
                                 self.heading_callback, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        if self.heading_source == 'map':
            self.create_subscription(String, '/visual_nav/diagnostics', self.diagnostics_cb, 10)
        if self.rate_compensation:
            self.create_subscription(Imu, '/mavros/imu/data', self.imu_cb, qos)
            self.interval_cli = self.create_client(MessageInterval, '/mavros/set_message_interval')
            self.interval_set = False
            self.interval_timer = self.create_timer(2.0, self.request_attitude_rate)
        self.vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_enu', 10)
        self.body_vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_body', 10)
        self.get_logger().info(
            f'Optical flow started (hfov {self.hfov:.2f} rad, alpha {self.alpha}, '
            f'heading {self.heading_source}, rate compensation '
            f'{"roll %.3f pitch %.3f" % (self.k_roll, self.k_pitch) if self.rate_compensation else "off"}).')

    def alt_cb(self, msg):
        self.altitude = msg.data

    def heading_callback(self, msg):
        self.heading = msg.data

    def diagnostics_cb(self, msg):
        """Track the offset between the map (camera) heading and the compass."""
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if (self.heading is None or 'fix_heading_deg' not in d
                or d.get('inliers', 0) < self.heading_min_inliers):
            return
        sample = (d['fix_heading_deg'] - self.heading + 180.0) % 360.0 - 180.0
        if self.heading_offset is None:
            self.heading_offset = sample
            self.get_logger().info(f'Heading offset initialised from the map: {sample:+.1f} deg')
            return
        diff = (sample - self.heading_offset + 180.0) % 360.0 - 180.0
        if abs(diff) > self.heading_max_jump:
            return
        self.heading_offset += self.offset_gain * diff
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_offset_log > 10.0:
            self.last_offset_log = now
            self.get_logger().info(f'Heading offset (map - compass): {self.heading_offset:+.1f} deg')

    def imu_cb(self, msg):
        self.gyro.append((time.time(), msg.angular_velocity.x, msg.angular_velocity.y))

    def request_attitude_rate(self):
        """Ask the autopilot for ATTITUDE (id 30) fast enough to follow the body rates."""
        if self.interval_set:
            self.interval_timer.cancel()
            return
        if not self.interval_cli.service_is_ready():
            return
        future = self.interval_cli.call_async(
            MessageInterval.Request(message_id=30, message_rate=self.attitude_rate))

        def done(fut):
            try:
                if fut.result().success:
                    self.interval_set = True
                    self.get_logger().info(f'ATTITUDE stream set to {self.attitude_rate:.0f} Hz.')
            except Exception as e:
                self.get_logger().warn(f'Could not set the ATTITUDE rate: {e}')
        future.add_done_callback(done)

    def body_rates(self, window=0.1):
        """Mean (roll, pitch) rate over the last window seconds, or None."""
        now = time.time()
        recent = [(p, q) for t, p, q in self.gyro if now - t <= window]
        if not recent and self.gyro and now - self.gyro[-1][0] <= 0.5:
            recent = [self.gyro[-1][1:]]
        if not recent:
            return None
        return float(np.mean([r[0] for r in recent])), float(np.mean([r[1] for r in recent]))

    def flow_heading(self):
        """Heading used to rotate the flow into ENU."""
        if self.heading_source == 'map' and self.heading_offset is not None:
            return (self.heading + self.heading_offset) % 360.0
        return self.heading

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
            heading = self.flow_heading()
            vel = self.flow.update(gray, stamp, self.altitude, heading)
            if vel is None:
                return
            fwd, right = vel[2], vel[3]
            rates = self.body_rates() if self.rate_compensation else None
            if rates is not None:
                roll_rate, pitch_rate = rates
                right += self.k_roll * roll_rate * self.altitude
                fwd -= self.k_pitch * pitch_rate * self.altitude
            east, north = body_to_enu(fwd, right, heading)
            vel = (east, north, fwd, right)
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

#!/usr/bin/env python3
"""
Visual Estimator Node

Estimates the UAV position in GNSS-denied flight by matching camera frames
against a georeferenced map database (see build_map_db), and dead-reckons
between fixes with optical-flow velocity.

Localization itself lives in gnss_denied_nav.localization (ROS-free); this
node only feeds it frames, gates and blends its fixes, and publishes poses.
"""
import csv
import json
import math
import os
import threading
import time

import rclpy
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TwistStamped
from mavros_msgs.msg import State
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float64, String

from gnss_denied_nav import geo
from gnss_denied_nav.run_logging import resolve_run_dir
from gnss_denied_nav.localization import (
    CameraModel, Localizer, LocalizerConfig, MapDatabase)


class VisualEstimatorNode(Node):
    """Estimates UAV position by matching camera images to a map database."""

    def __init__(self):
        super().__init__('visual_estimator_node')
        self.is_landing = False
        self.parallel_cb_group = ReentrantCallbackGroup()
        # Only one frame is processed at a time; frames arriving meanwhile are dropped.
        self.frame_lock = threading.Lock()

        # --- Parameters ---
        self.declare_parameter('db_path', '')
        self.declare_parameter('start_lat', -35.3658674)
        self.declare_parameter('start_lon', 149.1652376)
        self.declare_parameter('camera_hfov', 2.0)
        self.declare_parameter('smoothing_alpha', 0.4)
        self.declare_parameter('min_altitude', 15.0)
        # Map frames matched per camera frame, nearest to the predicted position.
        self.declare_parameter('max_candidates', 3)
        # Stop at the first candidate (nearest first) whose fix has this many inliers; 0 = try all.
        self.declare_parameter('accept_inliers', 0)
        # Search radius around the predicted position; grows with time since the last fix.
        self.declare_parameter('search_radius', 40.0)
        self.declare_parameter('search_radius_growth', 2.0)   # m per second without a fix
        self.declare_parameter('max_search_radius', 150.0)
        # Start localized at the known take-off point (ENU origin) so the pose is
        # valid from the ground and the first database fix is gated like any other.
        self.declare_parameter('init_at_home', False)
        # Directory for this run's logs (set by the launch file; default ~/gnss_nav_logs/<time>).
        self.declare_parameter('run_dir', '')

        db_path = os.path.expanduser(self.get_parameter('db_path').value)
        self.start_lat = float(self.get_parameter('start_lat').value)
        self.start_lon = float(self.get_parameter('start_lon').value)
        self.SMOOTHING_ALPHA = float(self.get_parameter('smoothing_alpha').value)
        self.min_altitude = float(self.get_parameter('min_altitude').value)
        self.search_radius = float(self.get_parameter('search_radius').value)
        self.search_radius_growth = float(self.get_parameter('search_radius_growth').value)
        self.max_search_radius = float(self.get_parameter('max_search_radius').value)
        self.init_at_home = bool(self.get_parameter('init_at_home').value)

        if not db_path or not os.path.exists(db_path):
            self.get_logger().error(
                f"Map database not found at '{db_path}'. "
                "Set 'db_path' in visual_nav_params.yaml or the launch file.")
            raise FileNotFoundError(f"Database file not found: '{db_path}'")

        # --- Localizer ---
        self.get_logger().info(f'Loading map database {db_path} ...')
        db = MapDatabase.load(db_path)
        for note in db.notes:
            self.get_logger().warn(note)
        db.set_origin(self.start_lat, self.start_lon)
        w, h = db.image_size
        camera = CameraModel(w, h, float(self.get_parameter('camera_hfov').value))
        # The global index is only needed for a first fix without a prior.
        config = LocalizerConfig(
            max_candidates=int(self.get_parameter('max_candidates').value),
            accept_inliers=int(self.get_parameter('accept_inliers').value) or None,
            global_retrieval=not self.init_at_home)
        self.localizer = Localizer(db, camera, config)
        if db.image_based and not self.init_at_home:
            self.get_logger().error(
                f'{db.method} maps have no descriptors for global retrieval: without '
                'init_at_home there is no first fix. Use init_at_home (GUIDED launch).')
        self.algorithm = db.method
        self.get_logger().info(
            f'Map ready: {db.method}, {len(db)} frames, {len(db.kpts)} features.')

        # CSV log
        run_dir = resolve_run_dir(self.get_parameter('run_dir').value)
        self.csv_file = open(os.path.join(run_dir, 'visual_log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'source', 'x', 'y', 'inliers', 'sigma_m'])

        # --- State ---
        self.mavros_state = State()
        self.is_localized = self.init_at_home
        self.last_db_success_time = time.time()
        self.last_timeout_warn = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_vel_x = 0.0
        self.current_vel_y = 0.0
        self.last_vel_time = None
        self.current_altitude = 0.0
        # Autopilot compass heading: only a coarse prior for the localizer (turning
        # the query image for SuperPoint, rejecting fixes with a wrong heading).
        self.compass_heading = None
        self.compass_time = 0.0

        # --- Publishers & Subscriptions ---
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_metrics = self.create_publisher(String, '/visual_perf_metrics', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose_enu', 10)
        self.gps_publisher = self.create_publisher(NavSatFix, '/visual_gps', qos_sensor)
        # Per-frame localization details (JSON) for the GCS, loggers and debugging.
        self.diag_pub = self.create_publisher(String, '/visual_nav/diagnostics', 10)
        # Map frame positions, latched so late subscribers (GCS) still get them.
        qos_latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(PoseArray, '/visual_nav/map_frames', qos_latched)
        self.publish_map_frames()

        cbg = self.parallel_cb_group
        self.create_subscription(Image, '/camera/image', self.image_callback,
                                 qos_profile=qos_sensor, callback_group=cbg)
        self.create_subscription(TwistStamped, '/visual_velocity_enu', self.velocity_callback,
                                 qos_profile=qos_sensor, callback_group=cbg)
        self.create_subscription(State, '/mavros/state', self.state_cb,
                                 qos_profile=qos_sensor, callback_group=cbg)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb,
                                 qos_sensor, callback_group=cbg)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.compass_cb,
                                 qos_sensor, callback_group=cbg)

        self.bridge = CvBridge()

        # When the pose feeds the autopilot it must be a steady stream, independent
        # of the velocity callbacks (also on the ground and during landing).
        if self.init_at_home:
            self.create_timer(0.1, self.publish_current_pose, callback_group=cbg)

    def alt_cb(self, msg):
        self.current_altitude = msg.data

    def compass_cb(self, msg):
        self.compass_heading = msg.data
        self.compass_time = time.time()

    def state_cb(self, msg):
        self.mavros_state = msg

    def velocity_callback(self, msg):
        if self.mavros_state.mode == 'LAND':
            # No dead reckoning during landing (flow is unreliable near the ground).
            return
        # Integrate over the image-stamp interval (simulated time), not wall time.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = stamp - self.last_vel_time if self.last_vel_time is not None else 0.0
        self.last_vel_time = stamp
        if dt > 1.0 or dt < 0.0:
            dt = 0.0
        self.current_vel_x = msg.twist.linear.x
        self.current_vel_y = msg.twist.linear.y
        self.current_x += msg.twist.linear.x * dt
        self.current_y += msg.twist.linear.y * dt
        if not self.init_at_home:
            self.publish_current_pose()

    def image_callback(self, msg):
        if self.mavros_state.mode == 'LAND' or self.is_landing:
            if self.mavros_state.mode == 'LAND':
                self.is_landing = True
            return

        if self.current_altitude < self.min_altitude:
            self.last_db_success_time = time.time()
            return

        if not self.frame_lock.acquire(blocking=False):
            return  # previous frame still being processed

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.process_frame(gray, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')
        finally:
            self.frame_lock.release()

    def process_frame(self, gray, stamp):
        now = time.time()
        since_fix = now - self.last_db_success_time

        prior, radius = None, None
        if self.is_localized:
            prior = (self.current_x, self.current_y)
            radius = min(self.search_radius + self.search_radius_growth * since_fix,
                         self.max_search_radius)

        heading = self.compass_heading if now - self.compass_time < 2.0 else None
        fix, info = self.localizer.localize(gray, prior, radius, heading=heading)

        status, est_x, est_y = 'NO_MATCH', 0.0, 0.0
        if fix is not None and not self.is_localized:
            self.get_logger().warn(
                f'[{self.algorithm}] FIRST FIX: X={fix.x:.1f}, Y={fix.y:.1f} '
                f'({fix.n_inliers} inliers)')
            self.current_x, self.current_y = fix.x, fix.y
            self.is_localized = True
            self.last_db_success_time = now
            status, est_x, est_y = 'MATCH_INIT', fix.x, fix.y

        elif fix is not None:
            speed = math.hypot(self.current_vel_x, self.current_vel_y)
            acceptable_diff = speed * since_fix * 5.0 + 25.0
            dist_error = math.hypot(fix.x - self.current_x, fix.y - self.current_y)

            if dist_error < acceptable_diff:
                self.last_db_success_time = now
                a = self.SMOOTHING_ALPHA
                self.current_x = a * fix.x + (1 - a) * self.current_x
                self.current_y = a * fix.y + (1 - a) * self.current_y
                status, est_x, est_y = 'MATCH', self.current_x, self.current_y
                self.log_to_csv('DB_MATCH', self.current_x, self.current_y, fix)
            else:
                status, est_x, est_y = 'OUTLIER', fix.x, fix.y
                self.get_logger().warn(f'Outlier: {dist_error:.1f}m > {acceptable_diff:.1f}m')

        elif since_fix > 10.0 and now - self.last_timeout_warn > 5.0:
            self.last_timeout_warn = now
            self.get_logger().warn(
                f'No map fix for {since_fix:.0f}s: dead reckoning only '
                f'(search radius {radius or 0:.0f} m).')

        # Format: "ALGORITHM,STATUS,DURATION(ms),EST_X,EST_Y"
        self.pub_metrics.publish(String(
            data=f"{self.algorithm},{status},{info['ms_total']:.2f},{est_x:.2f},{est_y:.2f}"))
        self.publish_diagnostics(stamp, status, fix, info, radius)

    def publish_diagnostics(self, stamp, status, fix, info, radius):
        db = self.localizer.db
        nearest = float(np.min(np.hypot(db.enu[:, 0] - self.current_x,
                                        db.enu[:, 1] - self.current_y)))
        diag = {
            'stamp': stamp.sec + stamp.nanosec * 1e-9,
            'algorithm': self.algorithm,
            'status': status,
            'localized': self.is_localized,
            'x': self.current_x,
            'y': self.current_y,
            'since_fix_s': time.time() - self.last_db_success_time,
            'search_radius_m': radius,
            'nearest_map_frame_m': nearest,
            'n_features': info['n_features'],
            'n_candidates': len(info['candidates']),
            'rejects': info['rejects'],
            'rotations': info.get('rotations', []),
            'ms_extract': info['ms_extract'],
            'ms_retrieve': info['ms_retrieve'],
            'ms_match': info['ms_match'],
            'ms_total': info['ms_total'],
        }
        if fix is not None:
            ref = int(fix.ref_index)
            diag.update({
                'fix_x': fix.x, 'fix_y': fix.y, 'fix_heading_deg': fix.heading_deg,
                'sigma_m': fix.sigma_m, 'inliers': fix.n_inliers,
                'inlier_ratio': fix.inlier_ratio, 'scale': fix.scale,
                'ref_index': ref, 'ref_x': float(db.enu[ref, 0]), 'ref_y': float(db.enu[ref, 1]),
                'ref_file': str(db.filenames[ref]),
            })
        self.diag_pub.publish(String(data=json.dumps(diag)))

    def publish_map_frames(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        db = self.localizer.db
        for (x, y), hdg, alt in zip(db.enu, db.heading, db.rel_alt):
            p = Pose()
            p.position.x, p.position.y, p.position.z = float(x), float(y), float(alt)
            yaw = math.radians(90.0 - hdg)  # compass -> ENU yaw
            p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            msg.poses.append(p)
        self.map_pub.publish(msg)

    def publish_current_pose(self):
        stamp = self.get_clock().now().to_msg()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.current_x
        pose_msg.pose.position.y = self.current_y
        self.pose_pub.publish(pose_msg)

        lat, lon = geo.enu_to_latlon(self.current_x, self.current_y,
                                     self.start_lat, self.start_lon)
        gps_msg = NavSatFix()
        gps_msg.header.stamp = stamp
        gps_msg.header.frame_id = 'map'
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        self.gps_publisher.publish(gps_msg)

    def log_to_csv(self, source, x, y, fix):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([timestamp, source, x, y, fix.n_inliers, f'{fix.sigma_m:.2f}'])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualEstimatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

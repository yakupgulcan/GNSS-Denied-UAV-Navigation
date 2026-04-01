#!/usr/bin/env python3
"""
Visual Estimator Unified Node

Estimates UAV position in GNSS-denied environments by matching real-time
camera images against a pre-built feature database. Supports SIFT, ORB,
BRISK, AKAZE, and HOG algorithms, selectable via ROS2 parameters.

Includes a breadcrumb-based backtracking recovery mechanism for when
database matches are lost.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64, String, Bool
from mavros_msgs.msg import State
from cv_bridge import CvBridge
import math
import time
import csv
from datetime import datetime
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Algorithm implementations
from gnss_denied_nav.orb_detect_match import ORBDetectAndMatch
from gnss_denied_nav.brisk_detect_match import BRISKDetectAndMatch
from gnss_denied_nav.hog_detect_match import HOGDetectAndMatch
from gnss_denied_nav.sift_detect_match import SIFTDetectAndMatch
from gnss_denied_nav.akaze_detect_match import AKAZEDetectAndMatch


class VisualEstimatorNode(Node):
    """Estimates UAV position by matching camera images to a feature database."""

    # Supported algorithms and their constructor classes
    ALGORITHM_MAP = {
        "ORB": ORBDetectAndMatch,
        "BRISK": BRISKDetectAndMatch,
        "HOG": HOGDetectAndMatch,
        "SIFT": SIFTDetectAndMatch,
        "AKAZE": AKAZEDetectAndMatch,
    }

    def __init__(self):
        super().__init__('visual_estimator_node')
        self.is_landing = False
        self.parallel_cb_group = ReentrantCallbackGroup()

        # --- Declare ROS2 Parameters ---
        self.declare_parameter('algorithm', 'SIFT')
        self.declare_parameter('db_path', '')
        self.declare_parameter('start_lat', -35.3658674)
        self.declare_parameter('start_lon', 149.1652376)
        self.declare_parameter('smoothing_alpha', 0.4)
        self.declare_parameter('db_timeout', 12.0)
        self.declare_parameter('keyframe_interval', 1.0)
        self.declare_parameter('backtrack_speed', -1.5)
        self.declare_parameter('backtrack_skip_count', 5)
        self.declare_parameter('brake_duration', 2.0)
        self.declare_parameter('min_time_per_waypoint', 0.2)

        # --- Read Parameters ---
        self.algorithm = self.get_parameter('algorithm').get_parameter_value().string_value
        db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self.start_lat = self.get_parameter('start_lat').get_parameter_value().double_value
        self.start_lon = self.get_parameter('start_lon').get_parameter_value().double_value

        if not db_path:
            self.get_logger().error('db_path parameter is empty! Set it via config YAML or launch file.')
            raise ValueError('db_path parameter must be set.')

        if self.algorithm not in self.ALGORITHM_MAP:
            raise ValueError(f'Unknown algorithm: {self.algorithm}. '
                             f'Supported: {list(self.ALGORITHM_MAP.keys())}')

        # CSV Log
        ct = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f'visual_log_{self.algorithm}_{ct}.csv'
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'source', 'x', 'y', 'state'])

        # Initialize the selected algorithm
        self.get_logger().info(f'Loading database for {self.algorithm}...')
        AlgorithmClass = self.ALGORITHM_MAP[self.algorithm]
        if self.algorithm in ("ORB", "SIFT"):
            self.estimator = AlgorithmClass(db_path, n_features=1000)
        elif self.algorithm == "BRISK":
            self.estimator = AlgorithmClass(db_path, thresh=60)
        else:
            self.estimator = AlgorithmClass(db_path)
        self.get_logger().info('Database loaded.')

        # --- State Variables ---
        self.state = "NORMAL"  # NORMAL, BRAKING, BACKTRACKING
        self.mavros_state = State()
        self.is_localized = False

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_vel_x = 0.0
        self.current_vel_y = 0.0
        self.current_heading = 0.0
        self.last_vel_time = time.time()

        # --- Breadcrumb Parameters (from ROS2 params) ---
        self.last_db_success_time = time.time()
        self.DB_TIMEOUT = self.get_parameter('db_timeout').get_parameter_value().double_value
        self.KEYFRAME_INTERVAL = self.get_parameter('keyframe_interval').get_parameter_value().double_value
        self.last_keyframe_time = 0
        self.breadcrumb_stack = deque(maxlen=2000)

        # Backtrack settings
        self.MIN_TIME_PER_WAYPOINT = self.get_parameter('min_time_per_waypoint').get_parameter_value().double_value
        self.last_waypoint_switch_time = 0.0
        self.BACKTRACK_SPEED = self.get_parameter('backtrack_speed').get_parameter_value().double_value

        # Braking
        self.brake_start_time = 0.0
        self.BRAKE_DURATION = self.get_parameter('brake_duration').get_parameter_value().double_value
        self.BACKTRACK_SKIP_COUNT = self.get_parameter('backtrack_skip_count').get_parameter_value().integer_value

        # --- Publishers & Subscriptions ---
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub_metrics = self.create_publisher(String, '/visual_perf_metrics', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose_enu', 10)
        self.gps_publisher = self.create_publisher(NavSatFix, '/visual_gps', qos_sensor)
        self.pub_status = self.create_publisher(String, '/system/status', 10)
        self.pub_override = self.create_publisher(Bool, '/control/override', 10)
        self.pub_speed = self.create_publisher(Float64, '/control/target_speed', 10)
        self.pub_yaw = self.create_publisher(Float64, '/control/target_yaw', 10)

        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.vel_sub = self.create_subscription(TwistStamped, '/visual_velocity_enu', self.velocity_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.hdg_sub = self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.alt_sub = self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos_sensor, callback_group=self.parallel_cb_group)

        self.bridge = CvBridge()
        self.SMOOTHING_ALPHA = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        self.current_altitude = 0.0

    def alt_cb(self, msg): self.current_altitude = msg.data
    def hdg_cb(self, msg): self.current_heading = msg.data
    def state_cb(self, msg): self.mavros_state = msg

    def velocity_callback(self, msg):
        if self.mavros_state.mode == "LAND": return
        now = time.time()
        dt = now - self.last_vel_time
        self.last_vel_time = now
        if dt > 1.0: dt = 0.0 
        self.current_vel_x = msg.twist.linear.x
        self.current_vel_y = msg.twist.linear.y
        self.current_x += msg.twist.linear.x * dt
        self.current_y += msg.twist.linear.y * dt
        self.publish_current_pose() 

    def image_callback(self, msg):
        if self.mavros_state.mode == "LAND" or self.is_landing:
            if self.mavros_state.mode == "LAND": self.is_landing = True
            return
        
        if self.current_altitude < 15.0: 
            self.last_db_success_time = time.time() 
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.process_visual_data(cv_image)
            
        except Exception as e:
            self.get_logger().error(f'Img Callback Hata: {str(e)}')

    
    
    # --- MAIN PROCESSING LOOP ---
    def process_visual_data(self, cv_image):
        now = time.time()
        start_time = now  # Processing start time
        
        # Default values (to prevent errors)
        match_status = "NO_MATCH"
        est_x = 0.0
        est_y = 0.0
        
        # 1. RECORD BREADCRUMB
        if self.state == "NORMAL":
            if (now - self.last_keyframe_time) > self.KEYFRAME_INTERVAL:
                try:
                    ret = self.estimator.detectAndCompute(cv_image, None)
                    
                    valid_data = False
                    if self.algorithm == "HOG":
                        if ret is not None and ret[1] is not None: valid_data = True
                    else: 
                        if ret[1] is not None and len(ret[1]) > 5: valid_data = True
                    
                    if valid_data:
                        stored_data = ret[1] 
                        if self.algorithm == "HOG": stored_data = ret
                        
                        self.breadcrumb_stack.append({'data': stored_data, 'hdg': self.current_heading})
                        self.last_keyframe_time = now
                except Exception as e:
                    self.get_logger().error(f"Breadcrumb error: {e}")

        # 2. DATABASE MATCHING
        db_match_pos = self.try_get_db_position(cv_image)
        
        if db_match_pos is not None:
            match_x, match_y = db_match_pos
            
            # --- [A] FIRST FIX ---
            if not self.is_localized:
                self.get_logger().warn(f"[{self.algorithm}] FIRST FIX: X={match_x:.1f}, Y={match_y:.1f}")
                self.current_x = match_x
                self.current_y = match_y
                self.is_localized = True
                self.last_db_success_time = now
                # Record as MATCH_INIT on the very first lock
                match_status = "MATCH_INIT"
                est_x = match_x
                est_y = match_y
            
            else:
                # --- [B] NORMAL TRACKING ---
                elapsed_time = now - self.last_db_success_time
                current_speed = math.sqrt(self.current_vel_x**2 + self.current_vel_y**2)
                acceptable_diff = (current_speed * elapsed_time * 5.0) + 25.0 
                
                dist_error = math.sqrt((match_x - self.current_x)**2 + (match_y - self.current_y)**2)
                
                if dist_error < acceptable_diff:
                    self.last_db_success_time = now
                    self.current_x = self.SMOOTHING_ALPHA * match_x + (1 - self.SMOOTHING_ALPHA) * self.current_x
                    self.current_y = self.SMOOTHING_ALPHA * match_y + (1 - self.SMOOTHING_ALPHA) * self.current_y
                    
                    # SUCCESSFUL MATCH
                    match_status = "MATCH"
                    est_x = self.current_x
                    est_y = self.current_y
                    
                    if self.state != "NORMAL":
                        self.get_logger().info(f"[{self.algorithm}] RECOVERED! Normal mode.")
                        self.switch_to_normal_mode()

                    self.log_to_csv("DB_MATCH", self.current_x, self.current_y)
                else:
                    # OUTLIER
                    match_status = "OUTLIER"
                    est_x = match_x # Log what was found even if it's an outlier
                    est_y = match_y
                    self.get_logger().warn(f"Outlier: {dist_error:.1f}m > {acceptable_diff:.1f}m")

        # --- [C] NO MATCH ---
        else:
            time_since_last_match = now - self.last_db_success_time
            
            if self.state == "NORMAL":
                if time_since_last_match > self.DB_TIMEOUT:
                    self.get_logger().error(f"TIMEOUT! INITIATING BACKTRACK.")
                    self.switch_to_backtrack_mode()
            
            elif self.state == "BRAKING":
                self.send_control(0.0, self.current_heading)
                if (now - self.brake_start_time) > self.BRAKE_DURATION:
                    self.state = "BACKTRACKING"
                    for _ in range(min(self.BACKTRACK_SKIP_COUNT, len(self.breadcrumb_stack))):
                        self.breadcrumb_stack.pop()
                    self.last_waypoint_switch_time = now

            elif self.state == "BACKTRACKING":
                self.process_backtrack_logic(cv_image)
        
        # --- PUBLISH METRICS ---
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000.0
        
        metric_msg = String()
        # Format: "ALGORITHM,STATUS,DURATION(ms),EST_X,EST_Y"
        # Note: est_x/est_y could be 0.0 (NO_MATCH case), refer to status when analyzing
        metric_msg.data = f"{self.algorithm},{match_status},{duration_ms:.2f},{est_x:.2f},{est_y:.2f}"
        
        self.pub_metrics.publish(metric_msg)
        self.pub_status.publish(String(data=self.state))

    # --- YARDIMCI METOTLAR ---
    def try_get_db_position(self, cv_image):
        try:
            candidates = self.estimator.get_location(cv_image, min_match_count=6, top_k=5)
        except Exception: return None

        if not candidates: return None
        
        valid_points = []
        for gps_data, fname, score in candidates:
            if not isinstance(gps_data, dict): continue
            raw_lat = gps_data.get('lat') or gps_data.get('latitude')
            raw_lon = gps_data.get('lon') or gps_data.get('longitude')
            if raw_lat is None: continue
            
            x, y = self.latlon_to_enu(float(raw_lat), float(raw_lon))
            valid_points.append({'x': x, 'y': y, 'score': score})

        if not valid_points: return None

        best = valid_points[0]
        filtered = [p for p in valid_points if math.sqrt((p['x']-best['x'])**2 + (p['y']-best['y'])**2) < 50.0]
        if not filtered: return None

        sx=0; sy=0; sw=0
        for p in filtered:
            w = p['score']**2 
            sx += p['x']*w; sy += p['y']*w; sw += w
        
        if sw == 0: return None
        return (sx/sw, sy/sw)

    def process_backtrack_logic(self, cv_image):
        # Once stack is empty, it means we have returned to origin (home)
        if not self.breadcrumb_stack:
            self.get_logger().info("Breadcrumbs exhausted. Backtrack complete.")
            self.switch_to_normal_mode()
            self.last_db_success_time = time.time()
            return

        # Target: The top frame in the stack (the most recently added one)
        target_entry = self.breadcrumb_stack[-1]
        target_data = target_entry['data']
        target_hdg = target_entry['hdg']
        
        result = self.estimator.match_frame_to_descriptors(cv_image, target_data, min_match_count=6)
        
        if result:
            match_count, error_x, _, _, _, _ = result
            
            yaw_correction = error_x * 0.05 
            target_yaw = target_hdg + yaw_correction
            speed_cmd = self.BACKTRACK_SPEED
            
            now = time.time()
            dt = now - self.last_waypoint_switch_time
            
            # --- NEW LOGIC: NO STOPPING, KEEP GOING ---
            # If current frame matches the target sufficiently well (>35)
            # AND minimum time has elapsed since the last waypoint switch:
            if match_count > 35 and dt > self.MIN_TIME_PER_WAYPOINT:
                self.get_logger().info(f"Waypoint passed (Score: {match_count}). Moving to next.")
                
                # Pop current target from stack
                self.breadcrumb_stack.pop()
                
                # Reset timer
                self.last_waypoint_switch_time = now
                
                # Note: We don't stop the motors (no HOVER_WAIT).
                # The next cycle will automatically steer via the new target (stack[-1]).
                # For now, maintain current movement commands.
                self.send_control(speed_cmd, target_yaw)
                return 
            
            # Matched but we haven't quite reached the target yet (or min time hasn't passed)
            # Continue heading toward target
            self.send_control(speed_cmd, target_yaw)
            
        else:
            # If we momentarily lose lock on breadcrumbs, hold the heading and do not stop.
            self.get_logger().warn("Track momentarily lost, maintaining course...")
            self.send_control(0.0, target_hdg)

    def publish_current_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.current_x
        pose_msg.pose.position.y = self.current_y
        self.pose_pub.publish(pose_msg)
        
        final_lat, final_lon = self.enu_to_latlon(self.current_x, self.current_y)
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = "map"
        gps_msg.latitude = final_lat
        gps_msg.longitude = final_lon
        self.gps_publisher.publish(gps_msg)

    def send_control(self, speed, yaw):
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_yaw.publish(Float64(data=float(yaw)))

    def switch_to_backtrack_mode(self):
        self.state = "BRAKING"
        self.brake_start_time = time.time()
        self.pub_override.publish(Bool(data=True))
        
    def switch_to_normal_mode(self):
        self.state = "NORMAL"
        self.send_control(0.0, self.current_heading)
        self.pub_override.publish(Bool(data=False))

    def log_to_csv(self, source, x, y):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([timestamp, source, x, y, self.state])
        self.csv_file.flush()

    def latlon_to_enu(self, lat, lon):
        d_lat = lat - self.start_lat
        d_lon = lon - self.start_lon
        north = d_lat * 111132.0
        east = d_lon * (111132.0 * math.cos(math.radians(self.start_lat)))
        return east, north

    def enu_to_latlon(self, x, y):
        d_lat = y / 111132.0
        d_lon = x / (111132.0 * math.cos(math.radians(self.start_lat)))
        return self.start_lat + d_lat, self.start_lon + d_lon

    def destroy_node(self):
        if hasattr(self, 'csv_file'): self.csv_file.close()
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
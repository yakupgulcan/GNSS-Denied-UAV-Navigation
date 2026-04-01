#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OpticalFlowEstimator:
    def __init__(self, focal_length_px=370.0):
        self.focal_length = focal_length_px

        # Lucas-Kanade: larger window size reduces noise
        self.lk_params = dict(winSize=(31, 31),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.feature_params = dict(maxCorners=150,
                                   qualityLevel=0.15,
                                   minDistance=15,
                                   blockSize=7)

        self.prev_gray = None
        self.p0 = None
        self.prev_time = 0

    def estimate_velocity(self, current_img, height_m):
        if height_m < 0.3: return 0.0, 0.0

        curr_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        curr_time = time.time()

        # Re-detect features if tracking points are exhausted
        if self.prev_gray is None or self.p0 is None or len(self.p0) < 15:
            self.prev_gray = curr_gray
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            self.prev_time = curr_time
            return 0.0, 0.0

        dt = curr_time - self.prev_time
        if dt < 0.001: return 0.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.p0, None, **self.lk_params)

        vx, vy = 0.0, 0.0

        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            
            if len(good_new) > 0:
                flows = good_new - good_old

                # Use MEDIAN instead of mean to reduce jitter from outlier flows
                flow_x = np.median(flows[:, 0])
                flow_y = np.median(flows[:, 1])

                # Convert pixel flow to physical velocity (m/s)
                vx = (flow_x * height_m) / (self.focal_length * dt)
                vy = (flow_y * height_m) / (self.focal_length * dt)

            self.prev_gray = curr_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            self.prev_time = curr_time
        else:
            self.prev_gray = None
            
        return vx, vy

class VisualVelocityNode(Node):
    def __init__(self):
        super().__init__('visual_velocity_estimator')

        # --- Parameters ---

        # 1. Focal length (theoretical value, calibrate if needed)
        self.declare_parameter('focal_length', 370.0)

        # 2. Scale factor (calibration multiplier)
        # If ground truth speed is -5 but measured is -3 -> scale factor = 5/3 = 1.66
        self.declare_parameter('scale_factor', 1.66)

        # 3. Publish rate (Hz)
        self.declare_parameter('publish_rate', 10.0)

        # 4. Low-pass filter alpha (0.0 = max smoothing/lag, 1.0 = no filter)
        # Typical good range: 0.3 – 0.5
        self.declare_parameter('lpf_alpha', 0.2)

        f_len = self.get_parameter('focal_length').value
        self.scale_factor = self.get_parameter('scale_factor').value
        pub_rate = self.get_parameter('publish_rate').value
        self.alpha = self.get_parameter('lpf_alpha').value
        self.alpha_lateral = 0.1

        # Processing interval (seconds)
        self.process_interval = 1.0 / pub_rate
        self.last_process_time = 0

        self.estimator = OpticalFlowEstimator(focal_length_px=f_len)
        self.bridge = CvBridge()

        # State variables
        self.current_altitude = 0.0
        self.current_heading = 0.0
        self.is_heading_valid = False

        # Low-pass filter state
        self.prev_vn = 0.0
        self.prev_ve = 0.0
        self.prev_body_vx = 0.0
        self.prev_body_vy = 0.0

        qos_fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, '/camera/image', self.image_callback, qos_fast)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.heading_callback, qos_fast)

        self.vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_enu', 10)
        self.body_vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_body', 10)

        self.get_logger().info(f"Started. Scale: {self.scale_factor}, Alpha: {self.alpha}, Rate: {pub_rate}Hz")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)

    def alt_cb(self, msg: Float64):
        self.current_altitude = msg.data
    
    def heading_callback(self, msg: Float64):
        self.current_heading = msg.data
        self.is_heading_valid = True

    def image_callback(self, msg):
        # --- RATE LIMITER ---
        # Even if camera sends 60 FPS, we only process at the configured rate interval.
        now = time.time()
        if (now - self.last_process_time) < self.process_interval:
            return
        self.last_process_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            h = self.current_altitude
            if h < 0.5: h = 0.5 

            # 1. Raw pixel-space velocities
            v_px_x, v_px_y = self.estimator.estimate_velocity(cv_image, h)

            # 2. Convert to body frame
            # Camera mount convention: Flow Y(+) -> Forward, Flow X(-) -> Right
            v_body_fwd = v_px_y
            v_body_right = -v_px_x

            # Scale factor calibration correction
            v_body_fwd *= self.scale_factor
            v_body_right *= -self.scale_factor

            # Apply low-pass filter to body velocities
            v_body_x_filt = (self.alpha * v_body_fwd) + ((1.0 - self.alpha) * self.prev_body_vx)
            v_body_y_filt = (self.alpha_lateral * v_body_right) + ((1.0 - self.alpha_lateral) * self.prev_body_vy)
            
            self.prev_body_vx = v_body_x_filt
            self.prev_body_vy = v_body_y_filt

            if not self.is_heading_valid:
                return 

            # 3. Rotate body frame to ENU using current heading
            hdg_rad = math.radians(self.current_heading)
            
            v_north_filt = v_body_x_filt * math.cos(hdg_rad) + self.prev_body_vy * math.sin(hdg_rad)
            v_east_filt = v_body_y_filt * math.sin(hdg_rad) - self.prev_body_vy * math.cos(hdg_rad)

            """ 
            # Low-pass filter on ENU (alternative approach, currently unused):
            # new = alpha * raw + (1 - alpha) * old
            v_north_filt = (self.alpha * v_north_raw) + ((1.0 - self.alpha) * self.prev_vn)
            v_east_filt = (self.alpha * v_east_raw) + ((1.0 - self.alpha) * self.prev_ve)
            """
            
            # 4. Publish body-frame velocity
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = "map" 
            
            twist.twist.linear.x = float(v_body_x_filt)
            twist.twist.linear.y = float(v_body_y_filt)
            
            self.body_vel_pub.publish(twist)

            # 5. Publish ENU velocity
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = "map" 
            
            twist.twist.linear.x = float(v_east_filt)
            twist.twist.linear.y = float(v_north_filt)
            
            self.vel_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visual Navigation Planner

High-level mission planner that uses visual position estimates to navigate
the UAV along a line from start to target. Uses a lookahead-based line
following algorithm and publishes speed/yaw commands to the base controller.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, String
from mavros_msgs.msg import State, OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped


class VisualNavigationPlanner(Node):
    """Plans navigation along a line using visual position estimates."""

    def __init__(self):
        super().__init__('visual_navigation_planner')

        # --- Declare ROS2 Parameters ---
        self.declare_parameter('target_pos_x', 0.0)
        self.declare_parameter('target_pos_y', 550.0)
        self.declare_parameter('acceptance_radius', 30.0)
        self.declare_parameter('start_lat', -35.3658674)
        self.declare_parameter('start_lon', 149.1652376)
        self.declare_parameter('target_altitude', 30.0)
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('pos_p_gain', 0.5)
        self.declare_parameter('turn_threshold', 30.0)
        self.declare_parameter('lookahead_distance', 50.0)
        self.declare_parameter('kp_altitude', 17.0)
        self.declare_parameter('hover_pwm', 1500)
        self.declare_parameter('max_climb_pwm', 1750)
        self.declare_parameter('max_descend_pwm', 1250)
        self.declare_parameter('height_acceptance_error', 0.2)

        # --- Read Parameters ---
        self.target_pos_x = self.get_parameter('target_pos_x').get_parameter_value().double_value
        self.target_pos_y = self.get_parameter('target_pos_y').get_parameter_value().double_value
        self.acceptance_radius = self.get_parameter('acceptance_radius').get_parameter_value().double_value
        self.START_LAT = self.get_parameter('start_lat').get_parameter_value().double_value
        self.START_LON = self.get_parameter('start_lon').get_parameter_value().double_value
        self.TARGET_ALTITUDE = self.get_parameter('target_altitude').get_parameter_value().double_value
        self.MAX_SPEED = self.get_parameter('max_speed').get_parameter_value().double_value
        self.POS_P_GAIN = self.get_parameter('pos_p_gain').get_parameter_value().double_value
        self.TURN_THRESHOLD = self.get_parameter('turn_threshold').get_parameter_value().double_value
        self.LOOKAHEAD_DIST = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.KP_ALTITUDE = self.get_parameter('kp_altitude').get_parameter_value().double_value
        self.HOVER_PWM = self.get_parameter('hover_pwm').get_parameter_value().integer_value
        self.MAX_CLIMB_PWM = self.get_parameter('max_climb_pwm').get_parameter_value().integer_value
        self.MAX_DESCEND_PWM = self.get_parameter('max_descend_pwm').get_parameter_value().integer_value
        self.HEIGHT_ACCEPTANCE_ERROR = self.get_parameter('height_acceptance_error').get_parameter_value().double_value

        # State variables
        self.current_altitude = 0.0
        self.takeoff_achieved = False
        self.is_target_reached = False
        self.current_pos = {"x": 0.0, "y": 0.0}
        self.current_heading = 0.0
        self.mavros_state = State()
        self.mission_state = 0  # 0: Wait, 1: Takeoff, 2: Navigate, 3: Land, 4: Done
        self.current_vel_enu = {"x": 0.0, "y": 0.0}
        self.curent_vel_fwd = 0.0
        self.system_status = "NORMAL"

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Subscriptions ---
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(NavSatFix, '/visual_gps', self.visual_pose_cb, qos)
        self.create_subscription(PoseStamped, '/visual_pose_enu', self.visual_pose__enu_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.create_subscription(TwistStamped, '/visual_velocity_enu', self.vel_cb, qos)
        self.create_subscription(String, '/system/status', self.system_status_cb, qos)

        # --- Publishers ---
        self.pub_target_speed = self.create_publisher(Float64, '/control/target_speed', 10)
        self.pub_target_yaw = self.create_publisher(Float64, '/control/target_yaw', 10)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)

        # Services
        self.client_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.client_mode = self.create_client(SetMode, '/mavros/set_mode')

        self.create_timer(0.1, self.planner_loop)  # 10 Hz
        self.get_logger().info("Planner started. Waiting for base controller...")

    # --- Callbacks ---
    def alt_cb(self, msg): self.current_altitude = msg.data
    def state_cb(self, msg): self.mavros_state = msg
    def hdg_cb(self, msg): self.current_heading = msg.data

    def vel_cb(self, msg):
        self.current_vel_enu["x"] = msg.twist.linear.x  # East
        self.current_vel_enu["y"] = msg.twist.linear.y  # North

    def visual_pose__enu_cb(self, msg):
        self.current_pos["y"] = msg.pose.position.y
        self.current_pos["x"] = msg.pose.position.x

    def visual_pose_cb(self, msg):
        """Unused: GPS-based position fallback (kept for reference)."""
        if math.isnan(msg.latitude):
            return
        # GPS -> local ENU (flat-earth approximation, sufficient for small areas)
        d_lat = msg.latitude - self.START_LAT
        d_lon = msg.longitude - self.START_LON
        self.current_pos["y"] = d_lat * 111132.0  # North
        self.current_pos["x"] = d_lon * (111132.0 * math.cos(math.radians(self.START_LAT)))  # East

    def system_status_cb(self, msg):
        self.system_status = msg.data

    # --- Main Planning Loop ---
    def planner_loop(self):
        if not self.mavros_state.connected:
            return

        if self.system_status != "NORMAL":
            self.get_logger().info("System not in NORMAL mode, planner paused.")
            return

        if self.mission_state == 0:  # Takeoff
            self.takeoff()
            if self.takeoff_achieved:
                self.mission_state = 2
                if self.mavros_state.mode != "ALT_HOLD":
                    self.set_mode("ALT_HOLD")
                    self.send_rc(1500, 1500, self.HOVER_PWM, 1500)

        elif self.mission_state == 2:  # Navigate
            self.calculate_and_send_commands()

        elif self.mission_state == 3:  # Land
            speed = math.sqrt(self.current_vel_enu['x']**2 + self.current_vel_enu['y']**2)
            if speed > 1:
                for i in range(10):
                    self.publish_commands(0, self.current_heading)
                    self.get_logger().info("Zeroing speed before landing.")
                return
            if self.mavros_state.mode != "LAND":
                self.set_mode("LAND")
                # Zero speed/yaw commands so base controller does not interfere during descent
                self.publish_commands(0.0, self.current_heading)
            else:
                if not self.mavros_state.armed:
                    self.mission_state = 4
                    self.get_logger().info("Mission complete.")

    # --- Navigation Logic ---

    def calculate_and_send_commands(self):
        # 1. Distance to final target (used for stopping)
        real_dx = self.target_pos_x - self.current_pos["x"]
        real_dy = self.target_pos_y - self.current_pos["y"]
        distance_to_final = math.sqrt(real_dx * real_dx + real_dy * real_dy)

        # 2. Virtual target (lookahead point on the route line)
        # Assumption: route is a straight line from Start to Target.
        # We select a point LOOKAHEAD_DIST ahead on that line.
        # Since the ideal path is X=0, the virtual target X equals target_pos_x.
        virtual_target_y = self.current_pos["y"] + self.LOOKAHEAD_DIST
        virtual_target_x = self.target_pos_x

        # If closer to the final target than lookahead distance, lock onto final target
        if virtual_target_y > self.target_pos_y:
            virtual_target_y = self.target_pos_y
            virtual_target_x = self.target_pos_x

        # 3. Yaw calculation (toward virtual target)
        dx_virt = virtual_target_x - self.current_pos["x"]
        dy_virt = virtual_target_y - self.current_pos["y"]
        bearing_rad = math.atan2(dx_virt, dy_virt)
        bearing_deg = math.degrees(bearing_rad)
        if bearing_deg < 0:
            bearing_deg += 360.0

        # 4. Heading error
        heading_error = bearing_deg - self.current_heading
        if heading_error > 180: heading_error -= 360
        if heading_error < -180: heading_error += 360

        # 5. Speed profile (P-control: slow down as we approach)
        desired_speed = distance_to_final * self.POS_P_GAIN
        desired_speed = min(desired_speed, self.MAX_SPEED)

        # Stop and rotate if heading error exceeds threshold
        if abs(heading_error) > self.TURN_THRESHOLD:
            desired_speed = 0.0

        # 6. Arrival check
        if distance_to_final < self.acceptance_radius:
            self.get_logger().info("TARGET REACHED! Switching to landing.")
            self.mission_state = 3
            desired_speed = 0.0

        # Cross Track Error = current_pos["x"] (deviation from X=0 line)
        self.get_logger().info(
            f"Dist: {distance_to_final:.1f}m | Spd: {desired_speed:.1f} | Yaw: {bearing_deg:.1f}"
        )
        self.publish_commands(desired_speed, bearing_deg)


    def publish_commands(self, speed, yaw):
        msg_spd = Float64()
        msg_spd.data = float(speed)
        
        msg_yaw = Float64()
        msg_yaw.data = float(yaw)

        self.pub_target_speed.publish(msg_spd)
        self.pub_target_yaw.publish(msg_yaw)

    # --- Helper Functions ---
    def send_rc(self, roll, pitch, throttle, yaw):
        msg = OverrideRCIn()
        msg.channels = [65535] * 18
        msg.channels[0] = int(roll)
        msg.channels[1] = int(pitch)
        msg.channels[2] = int(throttle)
        msg.channels[3] = int(yaw)
        self.rc_pub.publish(msg)

    def arm_vehicle(self):
        if not self.mavros_state.armed:
            self.client_arm.call_async(CommandBool.Request(value=True))

    def takeoff(self):
        if self.takeoff_achieved: return
        if not self.mavros_state.connected: return
        if self.mavros_state.mode != "STABILIZE":
            self.set_mode("STABILIZE")
            self.send_rc(1500, 1500, 1500, 1500)
            return
        if not self.mavros_state.armed:
            self.arm_vehicle()
            self.send_rc(1500, 1500, 1000, 1500)
            return
        throttle_pwm = self.calculate_throttle_command()
        self.send_rc(1500, 1500, throttle_pwm, 1500 )
    
    def calculate_throttle_command(self):
        error = self.TARGET_ALTITUDE - self.current_altitude
        if abs(error) < self.HEIGHT_ACCEPTANCE_ERROR:
            self.takeoff_achieved = True
            return self.HOVER_PWM
        target_pwm = self.HOVER_PWM + (error * self.KP_ALTITUDE)
        return int(max(self.MAX_DESCEND_PWM, min(self.MAX_CLIMB_PWM, target_pwm)))
    
    def handle_takeoff_request(self):
        """Unused: alternative takeoff using ALT_HOLD (kept for reference)."""
        # Altitude is managed by PID_ALT in base_controller; we only need to ARM in ALT_HOLD.
        if self.mavros_state.mode != "ALT_HOLD":
            self.set_mode("ALT_HOLD")
            time.sleep(0.5)  # Allow time for mode change

        if not self.mavros_state.armed and self.mavros_state.mode == "ALT_HOLD":
            self.get_logger().info("Arming...")
            self.client_arm.call_async(CommandBool.Request(value=True))

    def set_mode(self, mode):
        if self.mavros_state.mode != mode:
            req = SetMode.Request(custom_mode=mode)
            self.client_mode.call_async(req)

def main():
    rclpy.init()
    node = VisualNavigationPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Send stop command on shutdown
        node.publish_commands(0.0, node.current_heading)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
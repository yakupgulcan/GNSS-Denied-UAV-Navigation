#!/usr/bin/env python3
"""
Base Controller (Velocity & Yaw PID)

Low-level controller that converts speed/yaw targets from the navigation
planner into RC PWM commands for ArduPilot via MAVROS. Uses PID controllers
for forward velocity, lateral drift correction, yaw, and altitude hold.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64, String
from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import CommandBool, SetMode


class PIDController:
    """Simple PID controller with anti-windup."""

    def __init__(self, kp, ki, kd, max_out, integrator_limit=100):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.i_limit = integrator_limit
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            return 0.0

        p_term = self.kp * error

        # Anti-windup: only integrate when error is small
        if abs(error) < 1.5:
            self.integral += error * dt
            self.integral = max(min(self.integral, self.i_limit), -self.i_limit)
        else:
            self.integral = 0.0

        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt

        output = p_term + i_term + d_term
        output = max(min(output, self.max_out), -self.max_out)

        self.prev_error = error
        self.last_time = current_time
        return output


class VelocityYawController(Node):
    """Converts speed/yaw commands to RC PWM outputs via PID control."""

    def __init__(self):
        super().__init__('velocity_yaw_controller')

        # --- Declare ROS2 Parameters ---
        self.declare_parameter('target_altitude', 30.0)
        self.declare_parameter('hover_pwm', 1500)
        self.declare_parameter('brake_gain', 120.0)
        self.declare_parameter('stop_threshold', 0.2)
        self.declare_parameter('pid_forward_kp', 50.0)
        self.declare_parameter('pid_forward_ki', 2.0)
        self.declare_parameter('pid_forward_kd', 5.0)
        self.declare_parameter('pid_lateral_kp', 150.0)
        self.declare_parameter('pid_lateral_ki', 5.0)
        self.declare_parameter('pid_lateral_kd', 10.0)
        self.declare_parameter('pid_yaw_kp', 20.0)
        self.declare_parameter('pid_yaw_ki', 0.6)
        self.declare_parameter('pid_yaw_kd', 0.5)
        self.declare_parameter('pid_alt_kp', 20.0)
        self.declare_parameter('pid_alt_ki', 2.0)
        self.declare_parameter('pid_alt_kd', 5.0)

        # --- Read Parameters ---
        self.TARGET_ALTITUDE = self.get_parameter('target_altitude').get_parameter_value().double_value
        self.HOVER_PWM = self.get_parameter('hover_pwm').get_parameter_value().integer_value
        self.BRAKE_GAIN = self.get_parameter('brake_gain').get_parameter_value().double_value
        self.STOP_THRESHOLD = self.get_parameter('stop_threshold').get_parameter_value().double_value

        # PID Controllers
        fwd_kp = self.get_parameter('pid_forward_kp').get_parameter_value().double_value
        fwd_ki = self.get_parameter('pid_forward_ki').get_parameter_value().double_value
        fwd_kd = self.get_parameter('pid_forward_kd').get_parameter_value().double_value
        lat_kp = self.get_parameter('pid_lateral_kp').get_parameter_value().double_value
        lat_ki = self.get_parameter('pid_lateral_ki').get_parameter_value().double_value
        lat_kd = self.get_parameter('pid_lateral_kd').get_parameter_value().double_value
        yaw_kp = self.get_parameter('pid_yaw_kp').get_parameter_value().double_value
        yaw_ki = self.get_parameter('pid_yaw_ki').get_parameter_value().double_value
        yaw_kd = self.get_parameter('pid_yaw_kd').get_parameter_value().double_value
        alt_kp = self.get_parameter('pid_alt_kp').get_parameter_value().double_value
        alt_ki = self.get_parameter('pid_alt_ki').get_parameter_value().double_value
        alt_kd = self.get_parameter('pid_alt_kd').get_parameter_value().double_value

        self.pid_vel_forward = PIDController(kp=fwd_kp, ki=fwd_ki, kd=fwd_kd, max_out=200)
        self.pid_vel_lateral = PIDController(kp=lat_kp, ki=lat_ki, kd=lat_kd, max_out=200, integrator_limit=100)
        self.pid_yaw = PIDController(kp=yaw_kp, ki=yaw_ki, kd=yaw_kd, max_out=100)
        self.pid_alt = PIDController(kp=alt_kp, ki=alt_ki, kd=alt_kd, max_out=250)

        # State variables
        self.curent_vel_fwd = 0.0
        self.current_vel_left = 0.0
        self.current_heading = 0.0
        self.current_alt = 0.0
        self.mavros_state = State()

        self.target_speed_body_x = 0.0
        self.target_yaw_deg = 0.0
        self.last_command_time = time.time()

        self.is_landing_sequence = False
        self.last_loop_time = time.time()
        self.LANDING_DESCENT_RATE = 10.0

        # QoS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subscriptions
        self.create_subscription(TwistStamped, '/visual_velocity_body', self.vel_body_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(Float64, '/control/target_speed', self.target_speed_cb, 10)
        self.create_subscription(Float64, '/control/target_yaw', self.target_yaw_cb, 10)
        self.create_subscription(String, '/system/status', self.status_cb, 10)

        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.client_mode = self.create_client(SetMode, '/mavros/set_mode')

        self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.get_logger().info("Velocity & Yaw Controller ready.")

    # --- CALLBACKLER ---
    def set_mode(self, mode):
        if self.mavros_state.mode != mode:
            req = SetMode.Request(custom_mode=mode)
            self.client_mode.call_async(req)

    def vel_body_cb(self, msg):
        self.curent_vel_fwd = msg.twist.linear.x 
        self.current_vel_left = msg.twist.linear.y 

    def hdg_cb(self, msg): self.current_heading = msg.data
    def alt_cb(self, msg): self.current_alt = msg.data
    def state_cb(self, msg): self.mavros_state = msg

    def target_speed_cb(self, msg):
        self.target_speed_body_x = msg.data
        self.last_command_time = time.time()

    def target_yaw_cb(self, msg):
        self.target_yaw_deg = msg.data
        self.last_command_time = time.time()
        
    def status_cb(self, msg):
        # Start landing sequence if a "LANDING" status message is received
        if msg.data == "LANDING" and not self.is_landing_sequence:
            self.get_logger().info("Landing sequence initiated...")
            self.is_landing_sequence = True

    # --- Braking ---
    def apply_braking_force(self, current_fwd_vel):
        if abs(current_fwd_vel) < self.STOP_THRESHOLD:
            return 1500
        brake_force = int(current_fwd_vel * self.BRAKE_GAIN)
        brake_pwm = 1500 + brake_force
        return max(1100, min(1900, brake_pwm))

    # --- Main Control Loop ---
    def control_loop(self):
        now = time.time()
        dt = now - self.last_loop_time
        self.last_loop_time = now

        if not self.mavros_state.connected:
            return

        # --- 1. Landing Sequence ---
        if self.mavros_state.mode == "LAND" or self.is_landing_sequence:
            self.is_landing_sequence = True  # Once entered, stay in landing
            self.target_speed_body_x = 0.0
            self.set_mode("ALT_HOLD")

        # --- 2. Safety timeout ---
        # If no new command received for 2 seconds and not landing, stop.
        elif (time.time() - self.last_command_time) > 2.0:
            self.target_speed_body_x = 0.0

        # --- 3. Coordinate system ---
        # Body-frame velocity comes directly from the sensor; no conversion needed.
        current_body_forward = self.curent_vel_fwd
        # NOTE: Sensors typically report +Y as Left. ArduPilot Roll: +PWM tilts Right.
        # If sensor +Y is Left and drone drifts left, we must roll Right (+Roll PWM).
        # Verify this if the drone moves opposite to the correction direction (e.g., reversed mount).
        current_body_lateral = self.current_vel_left

        # --- 4. PID Calculations ---

        # A) PITCH (Forward / Backward)
        rc_pitch = 1500
        if self.target_speed_body_x == 0.0 and abs(self.curent_vel_fwd) > 0.5:
            # Stop command while moving: apply braking
            rc_pitch = self.apply_braking_force(current_body_forward)
            self.pid_vel_forward.integral = 0.0
        else:
            # Normal PID
            error_forward = self.target_speed_body_x - current_body_forward
            out_forward = self.pid_vel_forward.update(error_forward)
            rc_pitch = 1500 - int(out_forward)

        # B) ROLL (Lateral) — zeroing logic
        # Target lateral speed is always 0 (no sideways drift)
        error_lateral = 0.0 - current_body_lateral

        # Deadzone: ignore very small errors to prevent oscillation
        if abs(error_lateral) < 0.1:
            out_lateral = 0
            # Keep integral so wind disturbance is still corrected
        else:
            out_lateral = self.pid_vel_lateral.update(error_lateral)

        # Roll PWM sign convention (ArduPilot):
        # PWM < 1500: tilt Left  |  PWM > 1500: tilt Right
        # If drone drifts Left (+Y), we need to tilt Right (PWM > 1500).
        # Error = 0 - (+Y) = negative  →  PID output negative
        # rc_roll = 1500 - out  →  1500 - (negative) = > 1500  →  tilt Right ✓
        rc_roll = 1500 - int(out_lateral)

        # C) YAW
        error_yaw = self.target_yaw_deg - self.current_heading
        if error_yaw > 180: error_yaw -= 360
        if error_yaw < -180: error_yaw += 360
        out_yaw = self.pid_yaw.update(error_yaw)
        rc_yaw = 1500 + int(out_yaw)

        # D) THROTTLE (Altitude hold)
        error_alt = self.TARGET_ALTITUDE - self.current_alt
        out_throttle = self.pid_alt.update(error_alt)
        rc_throttle = self.HOVER_PWM + int(out_throttle)

        # --- 5. Clamp and Send ---
        rc_pitch = max(1300, min(1700, rc_pitch))
        rc_roll = max(1200, min(1800, rc_roll))  # Slightly wider limit for lateral correction
        rc_yaw = max(1300, min(1700, rc_yaw))
        rc_throttle = max(1250, min(1750, rc_throttle))


        if self.is_landing_sequence:
            self.send_rc(rc_roll, rc_pitch, 1250, rc_yaw)
            if self.current_alt < 1.0:
                self.set_mode("LAND")
        elif self.mavros_state.mode == "ALT_HOLD":
             self.send_rc(rc_roll, rc_pitch, rc_throttle, rc_yaw)
        

    def send_rc(self, roll, pitch, throttle, yaw):
        msg = OverrideRCIn()
        msg.channels = [65535] * 18
        msg.channels[0] = int(roll)
        msg.channels[1] = int(pitch)
        msg.channels[2] = int(throttle)
        msg.channels[3] = int(yaw)
        self.rc_pub.publish(msg)

def main():
    rclpy.init()
    node = VelocityYawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_rc(1500, 1500, 1500, 1500)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
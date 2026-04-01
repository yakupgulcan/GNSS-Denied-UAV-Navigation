#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.msg import OverrideRCIn, State
from std_msgs.msg import Float64, String
from mavros_msgs.srv import CommandBool, SetMode
import time

class AltHoldPController(Node):
    def __init__(self):
        super().__init__('althold_p_controller')

        # --- Settings ---
        self.TARGET_ALTITUDE = 29.0  # Target altitude in metres
        self.KP_ALTITUDE = 12.0      # P gain (error * Kp = PWM offset)
        self.HOVER_PWM = 1500        # Neutral throttle (hover)
        self.MAX_PWM = 1750          # Maximum climb PWM
        self.MIN_PWM = 1250          # Maximum descend PWM
        self.ACCEPTANCE_ERROR = 1.0  # ±1m acceptance band

        # State variables
        self.takeoff_achieved = False
        self.current_altitude = 0.0
        self.mavros_state = State()
        self.is_target_reached = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- Subscriptions ---
        # Altitude (barometer or lidar, relative altitude)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)

        # --- Publishers ---
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.status_pub = self.create_publisher(String, '/althold_ctrl/status', 10)

        # --- Services ---
        self.client_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.client_mode = self.create_client(SetMode, '/mavros/set_mode')

        # --- Control loop (20 Hz) ---
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"AltHold P-Controller started. Target: {self.TARGET_ALTITUDE}m")

    # --- CALLBACKS ---
    def alt_cb(self, msg: Float64):
        self.current_altitude = msg.data

    def state_cb(self, msg: State):
        self.mavros_state = msg

    # --- P-Controller ---
    def calculate_throttle_command(self):
        """
        Simple P-controller:
          error = target - current
          pwm   = 1500 + (error * Kp)
        """
        error = self.TARGET_ALTITUDE - self.current_altitude

        # Within acceptance band: hold altitude
        if abs(error) < self.ACCEPTANCE_ERROR:
            self.is_target_reached = True
            return self.HOVER_PWM

        # P calculation
        control_output = error * self.KP_ALTITUDE
        
        # Add to hover base
        target_pwm = self.HOVER_PWM + control_output

        # Saturation
        target_pwm = max(self.MIN_PWM, min(self.MAX_PWM, target_pwm))

        return int(target_pwm)

    # --- Main Control Loop ---
    def control_loop(self):
        if self.takeoff_achieved:
            self.set_mode("ALT_HOLD")
            return
        # Wait for MAVROS connection
        if not self.mavros_state.connected:
            return

        # 1. Ensure correct mode and arm state
        if self.mavros_state.mode != "STABILIZE":
            self.set_mode("STABILIZE")
            self.send_rc_override(1500)  # Stay neutral during mode change
            return

        if not self.mavros_state.armed:
            self.arm_vehicle()
            self.send_rc_override(1000)  # Cut throttle while arming
            return

        # 2. Climb with P-controller
        throttle_pwm = self.calculate_throttle_command()

        # 3. Publish RC override
        self.send_rc_override(throttle_pwm)

        status_msg = (f"Alt: {self.current_altitude:.2f}m | "
                      f"Err: {(self.TARGET_ALTITUDE - self.current_altitude):.2f}m | "
                      f"PWM: {throttle_pwm}")
        self.get_logger().info(status_msg)

        if self.is_target_reached:
            # Keep looping to hold the hover altitude
            self.takeoff_achieved = True

    # --- Helper methods ---
    def send_rc_override(self, throttle_pwm):
        msg = OverrideRCIn()
        msg.channels = [65535] * 18
        msg.channels[0] = 1500  # Roll
        msg.channels[1] = 1500  # Pitch
        msg.channels[2] = int(throttle_pwm)  # Throttle (P-controller output)
        msg.channels[3] = 1500  # Yaw
        self.rc_pub.publish(msg)

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.client_mode.call_async(req)

    def arm_vehicle(self):
        req = CommandBool.Request()
        req.value = True
        self.client_arm.call_async(req)

def main():
    rclpy.init()
    node = AltHoldPController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_rc_override(1500)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
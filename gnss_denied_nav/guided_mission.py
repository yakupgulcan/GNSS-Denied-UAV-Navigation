#!/usr/bin/env python3
"""
Guided Mission

Flies the GNSS-denied mission with ArduPilot's own controllers. The visual
estimate reaches the autopilot as a GPS_INPUT stream (see gps_input_bridge),
so this node only issues high-level commands:

    wait for EKF position -> GUIDED -> arm -> takeoff -> fly to target -> LAND

It replaces navigation_runner + base_controller (RC overrides in ALT_HOLD).
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from mavros_msgs.msg import State, MountControl, GlobalPositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

from gnss_denied_nav.geo import enu_to_latlon, latlon_to_enu

# GlobalPositionTarget type_mask: use position only (ignore vel, accel, yaw, yaw rate)
POSITION_ONLY_MASK = (
    GlobalPositionTarget.IGNORE_VX | GlobalPositionTarget.IGNORE_VY | GlobalPositionTarget.IGNORE_VZ
    | GlobalPositionTarget.IGNORE_AFX | GlobalPositionTarget.IGNORE_AFY | GlobalPositionTarget.IGNORE_AFZ
    | GlobalPositionTarget.IGNORE_YAW | GlobalPositionTarget.IGNORE_YAW_RATE
)


class GuidedMission(Node):
    """High-level GUIDED mission sequencer."""

    def __init__(self):
        super().__init__('guided_mission')

        self.declare_parameter('start_lat', -35.3766471)
        self.declare_parameter('start_lon', 149.1652375)
        self.declare_parameter('target_pos_x', 0.0)       # East of home (m)
        self.declare_parameter('target_pos_y', 2900.0)    # North of home (m)
        self.declare_parameter('target_altitude', 30.0)
        self.declare_parameter('acceptance_radius', 10.0)
        self.declare_parameter('settle_time', 5.0)        # hover after takeoff before leaving (s)
        self.declare_parameter('land_on_arrival', True)

        self.start_lat = float(self.get_parameter('start_lat').value)
        self.start_lon = float(self.get_parameter('start_lon').value)
        self.target_x = float(self.get_parameter('target_pos_x').value)
        self.target_y = float(self.get_parameter('target_pos_y').value)
        self.target_alt = float(self.get_parameter('target_altitude').value)
        self.acceptance_radius = float(self.get_parameter('acceptance_radius').value)
        self.settle_time = float(self.get_parameter('settle_time').value)
        self.land_on_arrival = bool(self.get_parameter('land_on_arrival').value)

        self.target_lat, self.target_lon = enu_to_latlon(
            self.target_x, self.target_y, self.start_lat, self.start_lon)

        self.fcu_state = State()
        self.altitude = 0.0
        self.ekf_fix = None          # FCU's own global position (EKF output)
        self.phase = 'WAIT_POSITION'
        self.phase_start = time.time()
        self.last_request = 0.0
        self.last_log = 0.0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.ekf_fix_cb, qos)

        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)
        self.setpoint_pub = self.create_publisher(
            GlobalPositionTarget, '/mavros/setpoint_raw/global', 10)

        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.create_timer(0.5, self.step)
        self.get_logger().info(
            f"Guided mission ready. Target ENU ({self.target_x:.0f}, {self.target_y:.0f}) "
            f"-> ({self.target_lat:.7f}, {self.target_lon:.7f}) at {self.target_alt:.0f} m.")

    # --- Callbacks ---
    def state_cb(self, msg): self.fcu_state = msg
    def alt_cb(self, msg): self.altitude = msg.data

    def ekf_fix_cb(self, msg: NavSatFix):
        if msg.status.status >= 0 and not math.isnan(msg.latitude):
            self.ekf_fix = msg

    # --- Helpers ---
    def set_phase(self, phase):
        self.get_logger().info(f"Mission phase: {self.phase} -> {phase}")
        self.phase = phase
        self.phase_start = time.time()
        self.last_request = 0.0

    def due(self, period):
        """Rate-limit service requests; True at most once per period."""
        now = time.time()
        if now - self.last_request >= period:
            self.last_request = now
            return True
        return False

    def call(self, client, request, what):
        if not client.service_is_ready():
            return
        future = client.call_async(request)

        def done(fut):
            try:
                res = fut.result()
                ok = getattr(res, 'success', getattr(res, 'mode_sent', False))
                if not ok:
                    self.get_logger().warn(f"{what} rejected by FCU (will retry).")
            except Exception as e:
                self.get_logger().error(f"{what} failed: {e}")

        future.add_done_callback(done)

    def set_mode(self, mode):
        self.call(self.mode_cli, SetMode.Request(custom_mode=mode), f"Mode {mode}")

    def point_gimbal_down(self):
        msg = MountControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.mode = 2
        msg.pitch = -90.0
        self.mount_pub.publish(msg)

    def publish_target(self):
        msg = GlobalPositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        msg.type_mask = POSITION_ONLY_MASK
        msg.latitude = self.target_lat
        msg.longitude = self.target_lon
        msg.altitude = self.target_alt
        self.setpoint_pub.publish(msg)

    def distance_to_target(self):
        east, north = latlon_to_enu(
            self.ekf_fix.latitude, self.ekf_fix.longitude, self.start_lat, self.start_lon)
        return math.hypot(self.target_x - east, self.target_y - north), east, north

    # --- State machine (2 Hz) ---
    def step(self):
        if not self.fcu_state.connected:
            return

        if self.phase == 'WAIT_POSITION':
            self.point_gimbal_down()
            if self.ekf_fix is None:
                if self.due(5.0):
                    self.get_logger().info("Waiting for the FCU position estimate (EKF using GPS_INPUT)...")
                return
            self.set_phase('ARM')

        elif self.phase == 'ARM':
            self.point_gimbal_down()
            if self.fcu_state.mode != 'GUIDED':
                if self.due(2.0):
                    self.set_mode('GUIDED')
            elif not self.fcu_state.armed:
                if self.due(3.0):
                    self.call(self.arming_cli, CommandBool.Request(value=True), "Arming")
            else:
                self.set_phase('TAKEOFF')

        elif self.phase == 'TAKEOFF':
            self.point_gimbal_down()
            if not self.fcu_state.armed and (time.time() - self.phase_start) > 5.0:
                self.get_logger().warn("Disarmed before takeoff, arming again.")
                self.set_phase('ARM')
            elif self.altitude >= self.target_alt - 1.0:
                self.set_phase('SETTLE')
            elif self.altitude < 1.0 and self.due(4.0):
                req = CommandTOL.Request()
                req.altitude = self.target_alt
                self.call(self.takeoff_cli, req, "Takeoff")

        elif self.phase == 'SETTLE':
            self.point_gimbal_down()
            if (time.time() - self.phase_start) > self.settle_time:
                self.set_phase('NAVIGATE')

        elif self.phase == 'NAVIGATE':
            if self.fcu_state.mode != 'GUIDED':
                if self.due(5.0):
                    self.get_logger().warn(
                        f"FCU left GUIDED (mode {self.fcu_state.mode}); mission paused.")
                return
            self.publish_target()
            dist, east, north = self.distance_to_target()
            if time.time() - self.last_log > 5.0:
                self.last_log = time.time()
                self.get_logger().info(
                    f"EKF position ENU ({east:.1f}, {north:.1f}) | distance to target {dist:.1f} m")
            if dist < self.acceptance_radius:
                self.get_logger().info("Target reached.")
                self.set_phase('LAND' if self.land_on_arrival else 'DONE')

        elif self.phase == 'LAND':
            if self.fcu_state.mode != 'LAND':
                if self.due(2.0):
                    self.set_mode('LAND')
            elif not self.fcu_state.armed:
                self.set_phase('DONE')

        elif self.phase == 'DONE':
            if self.due(30.0):
                self.get_logger().info("Mission complete.")


def main(args=None):
    rclpy.init(args=args)
    node = GuidedMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GPS Input Bridge

Feeds the visual position estimate to ArduPilot as a MAVLink GPS_INPUT
stream (GPS1_TYPE=14), so the autopilot's own EKF and position controller
can be used (GUIDED / AUTO / LOITER) instead of RC overrides.

Input : /visual_gps (sensor_msgs/NavSatFix) from the visual estimator
Output: /mavros/gps_input/gps_input (mavros_msgs/GPSINPUT)

Only position is sent. Velocity is flagged as "ignore", so the EKF must be
configured with EK3_SRC1_VELXY=0. If the estimate goes stale the stream
stops on purpose: ArduPilot then sees a GPS loss and its failsafes apply.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from mavros_msgs.msg import GPSINPUT
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from gnss_denied_nav.geo import unix_to_gps_week

# GPS_INPUT_IGNORE_FLAGS (MAVLink common)
IGNORE_VEL_HORIZ = 8
IGNORE_VEL_VERT = 16
IGNORE_SPEED_ACCURACY = 32


class GpsInputBridge(Node):
    """Republishes the visual estimate as a GPS_INPUT stream for the FCU."""

    def __init__(self):
        super().__init__('gps_input_bridge')

        self.declare_parameter('publish_rate', 5.0)       # ArduPilot expects a steady 5 Hz
        self.declare_parameter('stale_timeout', 1.0)      # stop sending if estimate is older (s)
        self.declare_parameter('origin_alt_msl', 584.0)   # home altitude AMSL (m)
        self.declare_parameter('horiz_accuracy', 4.0)     # reported 1-sigma (m); EKF aligns only if < 5
        self.declare_parameter('vert_accuracy', 3.0)
        self.declare_parameter('hdop', 1.0)
        self.declare_parameter('satellites_visible', 10)

        rate = float(self.get_parameter('publish_rate').value)
        self.stale_timeout = float(self.get_parameter('stale_timeout').value)
        self.origin_alt = float(self.get_parameter('origin_alt_msl').value)
        self.horiz_accuracy = float(self.get_parameter('horiz_accuracy').value)
        self.vert_accuracy = float(self.get_parameter('vert_accuracy').value)
        self.hdop = float(self.get_parameter('hdop').value)
        self.sats = int(self.get_parameter('satellites_visible').value)

        self.last_fix = None
        self.last_fix_time = 0.0
        self.rel_alt = 0.0
        self.was_stale = True

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(NavSatFix, '/visual_gps', self.fix_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.pub = self.create_publisher(GPSINPUT, '/mavros/gps_input/gps_input', 10)

        self.create_timer(1.0 / rate, self.publish_gps_input)

        # The MAVROS gps_input plugin throttles to its own 'gps_rate' (default 5 Hz).
        # Throttling a 5 Hz stream at 5 Hz drops messages at random, and ArduPilot
        # flags the GPS unhealthy when the average interval exceeds 215 ms. Raise
        # the plugin limit so this node alone sets the timing.
        self.mavros_param_cli = self.create_client(SetParameters, '/mavros/gps_input/set_parameters')
        self.throttle_raised = False
        self.throttle_timer = self.create_timer(2.0, self.raise_mavros_throttle)
        self.get_logger().info(
            f"GPS_INPUT bridge ready ({rate:.0f} Hz, hAcc {self.horiz_accuracy:.1f} m).")

    def raise_mavros_throttle(self):
        if self.throttle_raised:
            self.throttle_timer.cancel()
            return
        if not self.mavros_param_cli.service_is_ready():
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name='gps_rate',
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=50.0))]
        future = self.mavros_param_cli.call_async(req)

        def done(fut):
            try:
                if all(r.successful for r in fut.result().results):
                    self.throttle_raised = True
                    self.get_logger().info("MAVROS gps_input throttle raised to 50 Hz.")
            except Exception as e:
                self.get_logger().warn(f"Could not set MAVROS gps_rate: {e}")

        future.add_done_callback(done)

    def fix_cb(self, msg: NavSatFix):
        self.last_fix = msg
        self.last_fix_time = time.time()

    def alt_cb(self, msg: Float64):
        self.rel_alt = msg.data

    def publish_gps_input(self):
        now = time.time()
        stale = self.last_fix is None or (now - self.last_fix_time) > self.stale_timeout
        if stale != self.was_stale:
            if stale:
                self.get_logger().warn("Visual estimate is stale: GPS_INPUT stream stopped.")
            else:
                self.get_logger().info("Visual estimate received: GPS_INPUT stream running.")
            self.was_stale = stale
        if stale:
            return

        week, week_ms = unix_to_gps_week(now)

        msg = GPSINPUT()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.gps_id = 0
        msg.fix_type = GPSINPUT.GPS_FIX_TYPE_3D_FIX
        msg.ignore_flags = IGNORE_VEL_HORIZ | IGNORE_VEL_VERT | IGNORE_SPEED_ACCURACY
        msg.time_week = week
        msg.time_week_ms = week_ms
        msg.lat = int(round(self.last_fix.latitude * 1e7))
        msg.lon = int(round(self.last_fix.longitude * 1e7))
        msg.alt = float(self.origin_alt + self.rel_alt)
        msg.hdop = self.hdop
        msg.vdop = self.hdop
        msg.horiz_accuracy = self.horiz_accuracy
        msg.vert_accuracy = self.vert_accuracy
        msg.satellites_visible = self.sats
        msg.yaw = 0  # 0 = not provided
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GpsInputBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ARM -> TAKEOFF -> WAIT ALT -> ENU(x,y) -> GPS -> WP PUSH -> AUTO
ROS2 + MAVROS + ArduPilot
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mavros_msgs.msg import State, MountControl, Waypoint
from std_msgs.msg import Float64
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, WaypointPush

from math import cos, pi
import time

EARTH_RADIUS = 6378137.0  # meters


def enu_to_gps(x, y, lat0, lon0):
    dlat = (y / EARTH_RADIUS) * (180.0 / pi)
    dlon = (x / (EARTH_RADIUS * cos(lat0 * pi / 180.0))) * (180.0 / pi)
    return lat0 + dlat, lon0 + dlon


class ArduTakeoffNode(Node):
    def __init__(self):
        super().__init__('ardupilot_enu_mission')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state = State()
        self.altitude = 0.0

        # ---------- PARAMETERS ----------
        self.declare_parameter('home_lat', -35.3658674)
        self.declare_parameter('home_lon', 149.1652376)
        self.declare_parameter('flight_alt', 30.0)
        self.declare_parameter('width_left', 40.0)
        self.declare_parameter('width_right', 40.0)
        self.declare_parameter('forward_distance', 600.0)
        self.declare_parameter('backward_distance', 0.0)
        self.declare_parameter('corridor_spacing', 20.0)

        self.home_lat = float(self.get_parameter('home_lat').value)
        self.home_lon = float(self.get_parameter('home_lon').value)
        self.flight_alt = float(self.get_parameter('flight_alt').value)
        self.width_left = float(self.get_parameter('width_left').value)
        self.width_right = float(self.get_parameter('width_right').value)
        self.forward_distance = float(self.get_parameter('forward_distance').value)
        self.backward_distance = float(self.get_parameter('backward_distance').value)
        self.corridor_spacing = float(self.get_parameter('corridor_spacing').value)

        # Generate ENU Waypoints dynamically based on rectangular scan parameters
        self.enu_points = self.generate_grid_waypoints()

        # ---------- SUB / PUB ----------
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.alt_sub = self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)

        # ---------- SERVICES ----------
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.wp_push_cli = self.create_client(WaypointPush, '/mavros/mission/push')

    # ---------- CALLBACKS ----------
    def state_cb(self, msg):
        self.state = msg

    def alt_cb(self, msg):
        self.altitude = msg.data

    # ---------- HELPERS ----------
    def wait_for_connection(self, timeout=15.0):
        self.get_logger().info("Waiting for FCU...")
        start = self.get_clock().now()
        while not self.state.connected:
            if (self.get_clock().now() - start) > Duration(seconds=timeout):
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def set_mode(self, mode):
        self.mode_cli.wait_for_service()
        req = SetMode.Request(custom_mode=mode)
        fut = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().mode_sent

    def arm(self):
        self.arming_cli.wait_for_service()
        req = CommandBool.Request(value=True)
        fut = self.arming_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().success

    def takeoff(self):
        self.takeoff_cli.wait_for_service()
        req = CommandTOL.Request()
        req.altitude = self.flight_alt
        fut = self.takeoff_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().success

    def wait_alt(self, tol=0.3):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.altitude >= self.flight_alt - tol:
                return True

    def set_gimbal_down(self):
        msg = MountControl()
        msg.mode = 2
        msg.pitch = -90.0
        self.mount_pub.publish(msg)

    # ---------- MISSION ----------
    def generate_grid_waypoints(self):
        min_x = 0.0 if self.width_left == 0.0 else -abs(self.width_left)
        max_x = abs(self.width_right)
        min_y = 0.0 if self.backward_distance == 0.0 else -abs(self.backward_distance)
        max_y = abs(self.forward_distance)
        spacing = max(1.0, abs(self.corridor_spacing))

        num_lanes = int(round((max_x - min_x) / spacing)) + 1
        x_coords = [min_x + i * spacing for i in range(num_lanes)]
        # Ensure max_x is included if round-off excluded it
        if abs(x_coords[-1] - max_x) > 0.01 and x_coords[-1] < max_x:
            x_coords.append(max_x)

        points = []
        for i, x in enumerate(x_coords):
            if i % 2 == 0:
                points.append((x, min_y))
                points.append((x, max_y))
            else:
                points.append((x, max_y))
                points.append((x, min_y))

        # Return to origin (0, 0) at the end of survey
        points.append((0.0, 0.0))

        self.get_logger().info(
            f"Generated survey grid: X[{min_x:.1f}m to {max_x:.1f}m], Y[{min_y:.1f}m to {max_y:.1f}m], "
            f"spacing={spacing:.1f}m ({len(x_coords)} lanes, {len(points)} waypoints)"
        )
        return points

    def build_waypoints(self):
        wps = []

        # TAKEOFF WP
        wp0 = Waypoint()
        wp0.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp0.command = 22
        wp0.is_current = True
        wp0.autocontinue = True
        wp0.x_lat = self.home_lat
        wp0.y_long = self.home_lon
        wp0.z_alt = self.flight_alt
        wps.append(wp0)

        for x, y in self.enu_points:
            lat, lon = enu_to_gps(x, y, self.home_lat, self.home_lon)

            wp = Waypoint()
            wp.frame = Waypoint.FRAME_GLOBAL_REL_ALT
            wp.command = 16
            wp.autocontinue = True
            wp.x_lat = lat
            wp.y_long = lon
            wp.z_alt = self.flight_alt

            self.get_logger().info(f"ENU ({x},{y}) -> GPS ({lat:.7f},{lon:.7f})")
            wps.append(wp)

        return wps

    def upload_mission(self, wps):
        self.wp_push_cli.wait_for_service()
        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints = wps
        fut = self.wp_push_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().success


def main():
    rclpy.init()
    node = ArduTakeoffNode()

    if not node.wait_for_connection():
        return
    node.set_mode('GUIDED')
    node.arm()
    time.sleep(1)

    node.takeoff()
    node.wait_alt()
    node.set_gimbal_down()

    wps = node.build_waypoints()
    node.upload_mission(wps)

    node.set_mode('AUTO')
    node.get_logger().info("🚀 ENU mission started")

    rclpy.spin(node)


if __name__ == '__main__':
    main()

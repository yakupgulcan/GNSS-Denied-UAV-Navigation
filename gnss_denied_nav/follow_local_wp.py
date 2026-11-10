#!/usr/bin/env python3
"""
Autonomous takeoff -> Gimbal down -> Corridor Swath Survey -> AUTO mission
ROS2 + MAVROS + ArduPilot
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mavros_msgs.msg import State, MountControl, Waypoint, WaypointReached
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, Bool
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
        self.current_gps = None

        # ---------- PARAMETERS ----------
        # Corridor & Target parameters
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 3000.0)
        self.declare_parameter('corridor_width', 80.0)   # Total swath width in meters (+- 40m)
        self.declare_parameter('lane_spacing', 20.0)     # Spacing between survey lines
        self.declare_parameter('flight_alt', 30.0)

        # Home GPS reference
        self.declare_parameter('auto_home', True)
        self.declare_parameter('home_lat', -35.3766471)  # Default for map_original (3000m map)
        self.declare_parameter('home_lon', 149.1652374)

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.corridor_width = float(self.get_parameter('corridor_width').value)
        self.lane_spacing = float(self.get_parameter('lane_spacing').value)
        self.flight_alt = float(self.get_parameter('flight_alt').value)
        self.auto_home = bool(self.get_parameter('auto_home').value)
        self.home_lat = float(self.get_parameter('home_lat').value)
        self.home_lon = float(self.get_parameter('home_lon').value)

        # Survey completion & Return parameters
        self.declare_parameter('rtl_on_finish', True)
        self.rtl_on_finish = bool(self.get_parameter('rtl_on_finish').value)

        # Generate ENU Waypoints dynamically
        self.enu_points = self.generate_corridor_waypoints()

        # Mission tracking state
        self.survey_completed = False
        self.mission_completed = False
        # Last survey point is len(enu_points) - 1, since the final point is return to (0,0)
        self.last_survey_wp_seq = max(1, len(self.enu_points) - 1)
        self.total_wps_count = len(self.enu_points)

        # ---------- SUB / PUB ----------
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.alt_sub = self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_cb, qos)
        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)
        self.record_pub = self.create_publisher(Bool, '/dataset/record', 10)
        self.wp_reached_sub = self.create_subscription(
            WaypointReached, '/mavros/mission/reached', self.wp_reached_cb, 10
        )

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

    def gps_cb(self, msg):
        if msg.status.status >= 0:
            self.current_gps = msg

    def wp_reached_cb(self, msg: WaypointReached):
        seq = msg.wp_seq
        self.get_logger().info(f"Waypoint #{seq} reached.")

        # Check if the final survey waypoint of the corridor is reached
        if seq >= self.last_survey_wp_seq and not self.survey_completed:
            self.survey_completed = True
            self.get_logger().info(
                f"🎯 Final survey waypoint #{seq} reached! All corridor swaths completed."
            )
            self.get_logger().info("🛑 Disabling image recording for return flight...")
            self.publish_recording_state(False)
            self.get_logger().info("🛫 Returning to home origin (0.0, 0.0)...")

        # Check if return-to-home waypoint is reached
        if seq >= self.total_wps_count and not self.mission_completed:
            self.mission_completed = True
            self.get_logger().info("🛬 Arrived back at home origin (0.0, 0.0). Survey flight finished!")
            if self.rtl_on_finish:
                self.get_logger().info("Switching mode to RTL for autonomous landing...")
                self.set_mode_async('RTL')

    # ---------- HELPERS ----------
    def wait_for_connection(self, timeout=15.0):
        self.get_logger().info("Waiting for FCU...")
        start = self.get_clock().now()
        while not self.state.connected:
            if (self.get_clock().now() - start) > Duration(seconds=timeout):
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def wait_for_gps(self, timeout=20.0):
        self.get_logger().info("Waiting for GPS fix from FCU...")
        start = self.get_clock().now()
        while rclpy.ok() and self.current_gps is None:
            if (self.get_clock().now() - start) > Duration(seconds=timeout):
                self.get_logger().warning("GPS wait timeout! Using configured parameter coordinates.")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.auto_home and self.current_gps is not None:
            self.home_lat = self.current_gps.latitude
            self.home_lon = self.current_gps.longitude
            self.get_logger().info(
                f"Auto-captured Home GPS: Lat={self.home_lat:.7f}, Lon={self.home_lon:.7f}"
            )
            return True
        return False

    def set_mode(self, mode):
        self.mode_cli.wait_for_service()
        req = SetMode.Request(custom_mode=mode)
        fut = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result().mode_sent

    def set_mode_async(self, mode):
        """Asynchronously change flight mode while the node is actively spinning."""
        if not self.mode_cli.service_is_ready():
            self.get_logger().warning("Mode service not ready, cannot switch mode")
            return
        req = SetMode.Request(custom_mode=mode)
        fut = self.mode_cli.call_async(req)

        def done_cb(future):
            try:
                res = future.result()
                if res.mode_sent:
                    self.get_logger().info(f"Flight mode successfully changed to {mode}.")
                else:
                    self.get_logger().warning(f"Flight mode change to {mode} was rejected.")
            except Exception as e:
                self.get_logger().error(f"Mode change service error: {e}")

        fut.add_done_callback(done_cb)

    def publish_recording_state(self, active: bool):
        """Broadcast data recording state to sensor nodes."""
        msg = Bool()
        msg.data = active
        self.record_pub.publish(msg)

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
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.mode = 2
        msg.pitch = -90.0
        self.mount_pub.publish(msg)
        self.get_logger().info("Gimbal set downward (-90 deg).")

    # ---------- MISSION GENERATION ----------
    def generate_corridor_waypoints(self):
        """
        Generates survey waypoints covering a corridor around target_x from Y=0 to Y=target_y.
        If corridor_width <= 0: Single corridor mode along target_x.
        """
        spacing = max(1.0, abs(self.lane_spacing))
        total_w = abs(self.corridor_width)

        if total_w <= 0.0 or spacing >= total_w:
            # Single lane mode (e.g. only fly corridor at target_x)
            x_coords = [self.target_x]
        else:
            half_w = total_w / 2.0
            min_x = self.target_x - half_w
            max_x = self.target_x + half_w
            num_lanes = int(round((max_x - min_x) / spacing)) + 1
            x_coords = [min_x + i * spacing for i in range(num_lanes)]
            if abs(x_coords[-1] - max_x) > 0.01 and x_coords[-1] < max_x:
                x_coords.append(max_x)

        points = []
        for i, x in enumerate(x_coords):
            if i % 2 == 0:
                # Even lane: fly North (0 -> target_y)
                points.append((x, 0.0))
                points.append((x, self.target_y))
            else:
                # Odd lane: fly South (target_y -> 0)
                points.append((x, self.target_y))
                points.append((x, 0.0))

        # Return to home origin (0, 0)
        points.append((0.0, 0.0))

        self.get_logger().info(
            f"Generated corridor survey: Target=({self.target_x:.1f}, {self.target_y:.1f}), "
            f"Width={total_w:.1f}m ({len(x_coords)} lanes, {len(points)} waypoints)"
        )
        return points

    def build_waypoints(self):
        wps = []

        # TAKEOFF WP (Item 0 = Home in ArduPilot)
        wp0 = Waypoint()
        wp0.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp0.command = 22  # NAV_TAKEOFF
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
            wp.command = 16  # NAV_WAYPOINT
            wp.autocontinue = True
            wp.x_lat = lat
            wp.y_long = lon
            wp.z_alt = self.flight_alt

            self.get_logger().info(f"ENU ({x:.1f}, {y:.1f}) -> GPS ({lat:.7f}, {lon:.7f})")
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
    node.wait_for_gps()
    node.set_mode('GUIDED')
    node.arm()
    time.sleep(1)

    node.takeoff()
    node.wait_alt()
    node.set_gimbal_down()

    wps = node.build_waypoints()
    node.upload_mission(wps)

    node.set_mode('AUTO')
    node.get_logger().info("🚀 Corridor survey mission started in AUTO mode")
    node.publish_recording_state(True)

    rclpy.spin(node)


if __name__ == '__main__':
    main()

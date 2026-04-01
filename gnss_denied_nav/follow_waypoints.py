#!/usr/bin/env python3
"""
Autonomous takeoff + mission upload + AUTO mode start for ArduPilot using ROS2 + MAVROS.
Waits for actual takeoff before uploading mission.

"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.msg import State, MountControl, Waypoint
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, WaypointPush
import time


class ArduTakeoffNode(Node):
    def __init__(self):
        super().__init__('ardupilot_takeoff')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state = State()
        self.altitude = 0.0

        # Subscribers
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.alt_sub = self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)

        # Publishers
        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)

        # Services
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.waypoint_push_cli = self.create_client(WaypointPush, '/mavros/mission/push')

    # ---------- Callbacks ----------
    def state_cb(self, msg: State):
        self.state = msg

    def alt_cb(self, msg: Float64):
        self.altitude = msg.data

    # ---------- Helper Methods ----------
    def wait_for_connection(self, timeout_s: float = 15.0) -> bool:
        self.get_logger().info('Waiting for FCU connection...')
        start = self.get_clock().now()
        while rclpy.ok() and not self.state.connected:
            if (self.get_clock().now() - start) > Duration(seconds=timeout_s):
                self.get_logger().error('FCU connection timeout!')
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('FCU connected.')
        return True

    def wait_service(self, client, name: str, timeout_s: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().error(f'{name} service not available!')
            return False
        return True

    def set_mode(self, mode: str = 'GUIDED') -> bool:
        self.get_logger().info(f'Setting mode to {mode}...')
        if not self.wait_service(self.mode_cli, 'set_mode'):
            return False
        req = SetMode.Request(custom_mode=mode)
        fut = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        success = fut.result() and fut.result().mode_sent
        self.get_logger().info('Mode set successfully.' if success else 'Failed to set mode!')
        return success

    def arm(self, value: bool = True) -> bool:
        action = 'Arming' if value else 'Disarming'
        self.get_logger().info(f'{action} drone...')
        if not self.wait_service(self.arming_cli, 'arming'):
            return False
        req = CommandBool.Request(value=value)
        fut = self.arming_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        success = fut.result() and fut.result().success
        self.get_logger().info(f'{action} {"successful" if success else "failed"}')
        return success

    def takeoff(self, altitude: float = 3.0) -> bool:
        self.get_logger().info(f'Sending takeoff command to {altitude} m...')
        if not self.wait_service(self.takeoff_cli, 'takeoff'):
            return False
        req = CommandTOL.Request()
        req.altitude = altitude
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        fut = self.takeoff_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        success = fut.result() and fut.result().success
        self.get_logger().info('Takeoff command sent.' if success else 'Takeoff command failed!')
        return success

    def wait_until_altitude(self, target_alt: float, tol: float = 0.3, timeout_s: float = 30.0) -> bool:
        """Wait until drone reaches target altitude."""
        self.get_logger().info(f'Waiting to reach {target_alt:.1f} m altitude...')
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if abs(self.altitude - target_alt) <= tol or self.altitude >= target_alt - tol:
                self.get_logger().info(f'Altitude reached: {self.altitude:.2f} m ✅')
                return True
            if (time.time() - start) > timeout_s:
                self.get_logger().warning(f'Altitude wait timeout! Current: {self.altitude:.2f} m')
                return False
        return False

    def set_gimbal_down(self):
        msg = MountControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.mode = 2
        msg.pitch = -90.0
        msg.roll = 0.0
        msg.yaw = 0.0
        self.mount_pub.publish(msg)
        self.get_logger().info("Gimbal set downward.")

    # ---------- Mission ----------
    def load_waypoints_from_file(self, filename: str):
        waypoints = []
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('QGC') or line.strip() == '':
                continue
            cols = line.strip().split('\t')
            if len(cols) < 12:
                continue

            wp = Waypoint()
            wp.is_current = bool(int(cols[1]))
            wp.frame = int(cols[2])
            wp.command = int(cols[3])
            wp.param1 = float(cols[4])
            wp.param2 = float(cols[5])
            wp.param3 = float(cols[6])
            wp.param4 = float(cols[7])
            wp.x_lat = float(cols[8])
            wp.y_long = float(cols[9])
            wp.z_alt = float(cols[10])
            wp.autocontinue = bool(int(cols[11]))
            waypoints.append(wp)

        self.get_logger().info(f"Loaded {len(waypoints)} waypoints from {filename}")
        return waypoints

    def upload_mission(self, waypoints):
        if not self.wait_service(self.waypoint_push_cli, 'mission/push'):
            return False

        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints = waypoints

        self.get_logger().info('Uploading mission to FCU...')
        fut = self.waypoint_push_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)

        result = fut.result()
        if result and result.success:
            self.get_logger().info(f"Mission uploaded successfully ({result.wp_transfered} WPs)")
            return True
        else:
            self.get_logger().error('Mission upload failed!')
            return False


def main():
    rclpy.init()
    node = ArduTakeoffNode()

    try:
        if not node.wait_for_connection():
            return
        if not node.set_mode('GUIDED'):
            return
        if not node.arm(True):
            return
        time.sleep(1.0)

        # Step 1: Takeoff
        if not node.takeoff(10.0):
            return

        # Step 2: Wait for real altitude
        node.wait_until_altitude(3.0, tol=0.3, timeout_s=40.0)

        # Step 3: Adjust gimbal
        node.set_gimbal_down()

        # Step 4: Upload mission
        waypoints = node.load_waypoints_from_file('')
        if not node.upload_mission(waypoints):
            return

        # Step 5: AUTO mode
        if not node.set_mode('AUTO'):
            return

        node.get_logger().info("Mission started successfully ✅")

    except Exception as e:
        node.get_logger().error(f'Mission failed: {str(e)}')
    finally:
        node.get_logger().info('Shutting down node...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

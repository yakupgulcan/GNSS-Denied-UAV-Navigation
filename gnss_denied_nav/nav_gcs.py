#!/usr/bin/env python3
"""
Advanced Visual Navigation GUI (ROS 2, MAVROS, PySide6)
Features:
1. Tab: Camera and Gimbal
2. Tab: Flight Modes
3. Tab: Visual GPS Map and Telemetry (ENU transformed)
"""

import sys
import os
import time
import threading
import math
import numpy as np

# Matplotlib Backend Setup
os.environ["QT_API"] = "pyside6"

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ROS Messages
from mavros_msgs.msg import State, MountControl
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError

# PySide6 Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QGroupBox, QDoubleSpinBox, 
                               QFormLayout, QTabWidget, QGridLayout, QFrame)
from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# Matplotlib Imports
import matplotlib
matplotlib.use('QtAgg') 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------------------------------------------------------
# 1. ROS 2 Node (Visual Navigator Logic)
# ------------------------------------------------------------------
class VisualNavNode(Node):
    def __init__(self):
        super().__init__('visual_gui_node')
        
        # --- NAVIGATION SETTINGS ---
        self.target_x = 0.0
        self.target_y = 550.0
        self.acceptance_radius = 20.0
        
        # Start GPS Coordinate
        self.START_LAT = -35.3658674     # <-- START GPS
        self.START_LON = 149.1652376 
        
        # State Variables
        self.current_state = State()
        self.current_pos = {"x": 0.0, "y": 0.0} # ENU (Meters)
        self.current_vel = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_heading = 0.0
        self.bridge = CvBridge()

        # Callback Signals (for GUI)
        self.gui_image_signal = None
        self.gui_log_signal = None
        self.gui_status_signal = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # --- SUBSCRIPTIONS ---
        self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        self.create_subscription(Image, '/camera/image', self.image_cb, qos_profile)
        
        # Custom Subscriptions
        self.create_subscription(NavSatFix, '/visual_gps', self.visual_pose_cb, qos_profile)
        self.create_subscription(TwistStamped, '/visual_velocity_enu', self.visual_vel_cb, qos_profile)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos_profile)

        # --- PUBLISHERS & SERVICES ---
        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

    # --- CALLBACKS ---
    def state_cb(self, msg):
        self.current_state = msg
        if self.gui_status_signal: self.gui_status_signal.emit(msg)

    def image_cb(self, msg):
        if self.gui_image_signal:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.gui_image_signal.emit(cv_img)
            except CvBridgeError: pass

    def visual_pose_cb(self, msg: NavSatFix):
        """
        Converts incoming Lat/Lon to meters (ENU) relative to start coordinate.
        """
        if math.isnan(msg.latitude) or math.isnan(msg.longitude): return
        
        d_lat = msg.latitude - self.START_LAT
        d_lon = msg.longitude - self.START_LON
        
        # Simple planar projection (sufficient for small areas)
        north = d_lat * 111132.0
        east = d_lon * (111132.0 * math.cos(math.radians(self.START_LAT)))
        
        self.current_pos["x"] = east
        self.current_pos["y"] = north

    def visual_vel_cb(self, msg: TwistStamped):
        """
        Reads incoming velocity data.
        """
        self.current_vel["x"] = msg.twist.linear.x
        self.current_vel["y"] = msg.twist.linear.y
        self.current_vel["z"] = msg.twist.linear.z

    def hdg_cb(self, msg: Float64):
        """
        Reads compass heading.
        """
        self.current_heading = msg.data

    # --- CONTROL FUNCTIONS ---
    def send_gimbal_cmd(self, roll, pitch, yaw):
        msg = MountControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.mode = 2 
        msg.pitch = float(pitch)
        msg.roll = float(roll)
        msg.yaw = float(yaw)
        self.mount_pub.publish(msg)

    def set_mode(self, mode):
        req = SetMode.Request(custom_mode=mode)
        self.mode_cli.call_async(req)

# ------------------------------------------------------------------
# 2. Main GUI
# ------------------------------------------------------------------
class MainWindow(QMainWindow):
    # GUI Signals (Thread-Safe update)
    update_image_signal = Signal(np.ndarray)
    update_status_signal = Signal(State)

    def __init__(self, ros_node: VisualNavNode):
        super().__init__()
        self.node = ros_node
        self.setWindowTitle("Visual Navigation Control Center")
        self.resize(1100, 700)

        # Signal connections with node
        self.node.gui_image_signal = self.update_image_signal
        self.node.gui_status_signal = self.update_status_signal

        self.update_image_signal.connect(self.update_cam_display)
        self.update_status_signal.connect(self.update_status_bar)

        # Map History
        self.path_history_x = []
        self.path_history_y = []

        self.init_ui()

        # Telemetry and Map update timer (10 Hz)
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self.update_telemetry_and_map)
        self.telemetry_timer.start(100) 

    def init_ui(self):
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top Status Bar
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("background-color: #333; color: white; border-radius: 5px;")
        sf_layout = QHBoxLayout(self.status_frame)
        self.lbl_main_mode = QLabel("MODE: ---")
        self.lbl_main_arm = QLabel("ARM: ---")
        font = QFont("Arial", 12, QFont.Bold)
        self.lbl_main_mode.setFont(font); self.lbl_main_arm.setFont(font)
        sf_layout.addWidget(self.lbl_main_mode)
        sf_layout.addStretch()
        sf_layout.addWidget(self.lbl_main_arm)
        main_layout.addWidget(self.status_frame)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #C2C7CB; }")
        
        # 1. TAB: Camera & Gimbal
        self.tab_camera = QWidget()
        self.init_tab_camera()
        self.tabs.addTab(self.tab_camera, "Camera & Gimbal")

        # 2. TAB: Flight Modes
        self.tab_modes = QWidget()
        self.init_tab_modes()
        self.tabs.addTab(self.tab_modes, "Flight Modes")

        # 3. TAB: Visual GPS Map
        self.tab_map = QWidget()
        self.init_tab_map()
        self.tabs.addTab(self.tab_map, "Visual Map & Telemetry")

        main_layout.addWidget(self.tabs)

    # --------------------------------------------------------
    # TAB 1: Camera and Gimbal
    # --------------------------------------------------------
    def init_tab_camera(self):
        layout = QHBoxLayout(self.tab_camera)

        # Left: Video Feed
        self.lbl_video = QLabel("NO VIDEO")
        self.lbl_video.setMinimumSize(640, 480)
        self.lbl_video.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_video, stretch=3)

        # Right: Gimbal Controls
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        
        gb_gimbal = QGroupBox("Gimbal Control")
        form = QFormLayout()
        
        self.sp_roll = QDoubleSpinBox(); self.sp_roll.setRange(-90, 90)
        self.sp_pitch = QDoubleSpinBox(); self.sp_pitch.setRange(-90, 0); self.sp_pitch.setValue(-90)
        self.sp_yaw = QDoubleSpinBox(); self.sp_yaw.setRange(-180, 180)
        
        btn_gimbal = QPushButton("Send Angle")
        btn_gimbal.setMinimumHeight(40)
        btn_gimbal.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        btn_gimbal.clicked.connect(lambda: self.node.send_gimbal_cmd(
            self.sp_roll.value(), self.sp_pitch.value(), self.sp_yaw.value()))
        
        form.addRow("Roll:", self.sp_roll)
        form.addRow("Pitch:", self.sp_pitch)
        form.addRow("Yaw:", self.sp_yaw)
        gb_gimbal.setLayout(form)
        
        r_layout.addWidget(gb_gimbal)
        r_layout.addWidget(btn_gimbal)
        r_layout.addStretch()
        
        layout.addWidget(right_panel, stretch=1)

    # --------------------------------------------------------
    # TAB 2: Flight Modes
    # --------------------------------------------------------
    def init_tab_modes(self):
        layout = QGridLayout(self.tab_modes)
        modes = [
            ("STABILIZE", 0, 0), ("LOITER", 0, 1), ("POSHOLD", 0, 2),
            ("GUIDED", 1, 0), ("AUTO", 1, 1), ("RTL", 1, 2),
            ("LAND", 2, 0), ("BRAKE", 2, 1), ("THROW", 2, 2)
        ]
        
        for name, r, c in modes:
            btn = QPushButton(name)
            btn.setMinimumHeight(80)
            btn.setStyleSheet("""
                QPushButton { font-size: 16px; font-weight: bold; background-color: #e1e1e1; border-radius: 10px; }
                QPushButton:hover { background-color: #d4d4d4; }
                QPushButton:pressed { background-color: #b0b0b0; }
            """)
            if name == "LAND": btn.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
            if name == "RTL": btn.setStyleSheet("background-color: #f0ad4e; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;")
            
            btn.clicked.connect(lambda checked=False, n=name: self.node.set_mode(n))
            layout.addWidget(btn, r, c)

    # --------------------------------------------------------
    # TAB 3: Visual Map & Telemetry
    # --------------------------------------------------------
    def init_tab_map(self):
        layout = QHBoxLayout(self.tab_map)

        # LEFT: Telemetry Data (Speed, Heading, Position)
        data_panel = QGroupBox("Live Data")
        data_panel.setStyleSheet("font-weight: bold;")
        dp_layout = QVBoxLayout(data_panel)

        # Speed Box
        self.lbl_vel_x = QLabel("Vel X: 0.00 m/s")
        self.lbl_vel_y = QLabel("Vel Y: 0.00 m/s")
        self.lbl_vel_total = QLabel("Speed: 0.00 m/s")
        for l in [self.lbl_vel_x, self.lbl_vel_y, self.lbl_vel_total]:
            l.setStyleSheet("font-size: 14px; color: blue; padding: 5px; border: 1px solid #ddd;")
            dp_layout.addWidget(l)

        # Heading Box
        self.lbl_heading = QLabel("Heading: 0.0°")
        self.lbl_heading.setStyleSheet("font-size: 18px; color: darkgreen; padding: 10px; border: 2px solid green; background-color: #e8f5e9;")
        self.lbl_heading.setAlignment(Qt.AlignCenter)
        dp_layout.addWidget(self.lbl_heading)

        # Position Box
        self.lbl_pos_x = QLabel("Pos X: 0.0 m")
        self.lbl_pos_y = QLabel("Pos Y: 0.0 m")
        for l in [self.lbl_pos_x, self.lbl_pos_y]:
            l.setStyleSheet("font-size: 14px; color: #333; padding: 5px;")
            dp_layout.addWidget(l)

        dp_layout.addStretch()
        layout.addWidget(data_panel, stretch=1)

        # RIGHT: Matplotlib Map
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # Initial Map Settings
        self.ax.set_title("Visual GPS Navigation (ENU)")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.grid(True)
        self.ax.axis('equal')

        # Draw Objects
        # 1. Target Line (Blue): (0,0) -> (TargetX, TargetY)
        self.line_target, = self.ax.plot([], [], 'b--', linewidth=2, label='Target Path')
        # 2. Start Point (Green Square)
        self.ax.plot(0, 0, 'gs', markersize=8, label='Start')
        # 3. Target Point (Blue Cross)
        self.ax.plot(self.node.target_x, self.node.target_y, 'bx', markersize=10, markeredgewidth=2, label='Goal')
        # 4. Drone Path (Red Line)
        self.line_path, = self.ax.plot([], [], 'r-', alpha=0.6, linewidth=1)
        # 5. Drone Position (Red Dot)
        self.point_drone, = self.ax.plot([], [], 'ro', markersize=6, label='Drone')
        
        self.ax.legend(loc='upper right')
        
        layout.addWidget(self.canvas, stretch=3)

    # --------------------------------------------------------
    # UPDATE FUNCTIONS
    # --------------------------------------------------------
    def update_telemetry_and_map(self):
        # 1. Fetch data
        px = self.node.current_pos["x"]
        py = self.node.current_pos["y"]
        vx = self.node.current_vel["x"]
        vy = self.node.current_vel["y"]
        hdg = self.node.current_heading

        # 2. Update Left Panel
        self.lbl_vel_x.setText(f"Vel X: {vx:.2f} m/s")
        self.lbl_vel_y.setText(f"Vel Y: {vy:.2f} m/s")
        speed = math.sqrt(vx**2 + vy**2)
        self.lbl_vel_total.setText(f"Speed: {speed:.2f} m/s")
        self.lbl_heading.setText(f"HDG: {hdg:.1f}°")
        self.lbl_pos_x.setText(f"Pos X (East): {px:.1f} m")
        self.lbl_pos_y.setText(f"Pos Y (North): {py:.1f} m")

        # 3. Update Map Data
        # Append to history
        self.path_history_x.append(px)
        self.path_history_y.append(py)
        # Keep last 1000 points for performance
        if len(self.path_history_x) > 1000:
            self.path_history_x.pop(0)
            self.path_history_y.pop(0)

        # Target Line: (0,0) to (TargetX, TargetY)
        self.line_target.set_data([0, self.node.target_x], [0, self.node.target_y])
        
        # Drone Path and Position
        self.line_path.set_data(self.path_history_x, self.path_history_y)
        self.point_drone.set_data([px], [py])

        # Scale view (ensure both drone and target are visible)
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Draw
        self.canvas.draw_idle()

    @Slot(np.ndarray)
    def update_cam_display(self, cv_img):
        # OpenCV BGR -> Qt RGB
        img = cv_img.copy() # Safety copy
        h, w, ch = img.shape
        rgb_image = np.ascontiguousarray(img[..., ::-1]) # BGR to RGB
        qimg = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))

    @Slot(State)
    def update_status_bar(self, msg):
        self.lbl_main_mode.setText(f"MODE: {msg.mode}")
        self.lbl_main_arm.setText(f"ARM: {'YES' if msg.armed else 'NO'}")
        if msg.armed:
            self.lbl_main_arm.setStyleSheet("color: red;")
        else:
            self.lbl_main_arm.setStyleSheet("color: white;")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    rclpy.init()
    node = VisualNavNode()
    
    # Run ROS in a thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    app = QApplication(sys.argv)
    win = MainWindow(node)
    win.show()

    try:
        sys.exit(app.exec())
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
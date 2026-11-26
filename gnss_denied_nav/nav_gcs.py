#!/usr/bin/env python3
"""
Navigation Ground Control Station (ROS 2, MAVROS, PySide6, pyqtgraph)

Tabs:
  1. Navigation  - live comparison of the visual estimate against Gazebo ground
                   truth: map with map frames / true / estimated / EKF paths,
                   localization status, errors, speeds, altitude, matcher stats.
  2. Camera & Gimbal
  3. Flight Modes

Data sources:
  /odometry                          ground truth (Gazebo), relative to its first pose
  /visual_gps                        visual estimate (lat/lon)
  /visual_nav/diagnostics            per-frame localization details (JSON)
  /visual_nav/map_frames             map frame positions (latched)
  /visual_velocity_enu               optical-flow velocity
  /mavros/global_position/global     ArduPilot EKF position
  /mavros/local_position/velocity_local  ArduPilot EKF velocity
  /mavros/gps_input/gps_input        GPS_INPUT stream (rate only)
"""
import json
import math
import os
import sys
import threading
import time
from collections import deque

# pyqtgraph ignores QT_API and picks the first binding it finds (PyQt6/PyQt5 are
# also installed here); pin it to PySide6 and import PySide6 before pyqtgraph.
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import PySide6  # noqa: E402,F401
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge, CvBridgeError  # noqa: E402
from geometry_msgs.msg import PoseArray, TwistStamped  # noqa: E402
from mavros_msgs.msg import GPSINPUT, MountControl, State  # noqa: E402
from mavros_msgs.srv import CommandBool, SetMode  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from PySide6.QtCore import Qt, QTimer, Signal, Slot  # noqa: E402
from PySide6.QtGui import QFont, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image, NavSatFix  # noqa: E402
from std_msgs.msg import Float64, String  # noqa: E402

from gnss_denied_nav import geo  # noqa: E402

STATUS_COLORS = {'MATCH': '#2e7d32', 'MATCH_INIT': '#2e7d32', 'OUTLIER': '#ef6c00',
                 'NO_MATCH': '#c62828'}


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# ----------------------------------------------------------------------
# ROS node: collects the latest values; the GUI polls a snapshot.
# ----------------------------------------------------------------------
class GcsNode(Node):

    def __init__(self):
        super().__init__('visual_gui_node')
        self.declare_parameter('start_lat', -35.3766471)
        self.declare_parameter('start_lon', 149.1652375)
        self.declare_parameter('target_pos_x', 0.0)
        self.declare_parameter('target_pos_y', 2900.0)
        self.start_lat = float(self.get_parameter('start_lat').value)
        self.start_lon = float(self.get_parameter('start_lon').value)
        self.target = (float(self.get_parameter('target_pos_x').value),
                       float(self.get_parameter('target_pos_y').value))

        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.image_signal = None
        self.last_image_wall = 0.0

        self.state = State()
        self.heading = None
        self.sim_now = None
        # Ground truth
        self.gt_home = None
        self.gt = None                       # (e, n, u)
        self.gt_hist = deque(maxlen=50)      # (t, e, n) for velocity
        self.gt_vel = None                   # (ve, vn)
        # Estimates
        self.est = None                      # visual estimate (e, n)
        self.ekf = None                      # EKF position (e, n)
        self.ekf_vel = None
        self.ekf_alt = None
        self.flow_vel = None
        self.diag = {}
        self.diag_stamps = deque(maxlen=200)  # (sim t, accepted fix)
        self.gps_input_times = deque(maxlen=50)
        self.map_frames = None               # (N, 2)

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(State, '/mavros/state', self.state_cb, be)
        self.create_subscription(Image, '/camera/image', self.image_cb, be)
        self.create_subscription(Odometry, '/odometry', self.odom_cb, 20)
        self.create_subscription(NavSatFix, '/visual_gps', self.est_cb, be)
        self.create_subscription(String, '/visual_nav/diagnostics', self.diag_cb, 10)
        self.create_subscription(PoseArray, '/visual_nav/map_frames', self.map_cb, latched)
        self.create_subscription(TwistStamped, '/visual_velocity_enu', self.flow_cb, be)
        self.create_subscription(NavSatFix, '/mavros/global_position/global', self.ekf_cb, be)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local',
                                 self.ekf_vel_cb, be)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, be)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, be)
        self.create_subscription(GPSINPUT, '/mavros/gps_input/gps_input', self.gps_input_cb, 10)

        self.mount_pub = self.create_publisher(MountControl, '/mavros/mount_control/command', 10)
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

    def to_enu(self, lat, lon):
        return geo.latlon_to_enu(lat, lon, self.start_lat, self.start_lon)

    # --- callbacks
    def state_cb(self, msg):
        with self.lock:
            self.state = msg

    def hdg_cb(self, msg):
        self.heading = msg.data

    def alt_cb(self, msg):
        self.ekf_alt = msg.data

    def image_cb(self, msg):
        now = time.time()
        if self.image_signal is None or now - self.last_image_wall < 0.1:
            return
        self.last_image_wall = now
        try:
            self.image_signal.emit(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        except CvBridgeError:
            pass

    def odom_cb(self, msg):
        t = stamp_sec(msg.header.stamp)
        p = msg.pose.pose.position
        with self.lock:
            if self.gt_home is None:
                self.gt_home = (p.x, p.y, p.z)
            e, n, u = p.x - self.gt_home[0], p.y - self.gt_home[1], p.z - self.gt_home[2]
            self.gt = (e, n, u)
            self.sim_now = t
            self.gt_hist.append((t, e, n))
            # Velocity over ~0.3 s of poses (robust to the odometry frame convention).
            old = next((h for h in self.gt_hist if t - h[0] <= 0.3), None)
            if old is not None and t - old[0] > 0.05:
                dt = t - old[0]
                self.gt_vel = ((e - old[1]) / dt, (n - old[2]) / dt)

    def est_cb(self, msg):
        if math.isnan(msg.latitude):
            return
        with self.lock:
            self.est = self.to_enu(msg.latitude, msg.longitude)

    def ekf_cb(self, msg):
        # Before the EKF has a position, MAVROS publishes lat/lon 0 with
        # status NO_FIX (-1); drawing it would put a point thousands of km away.
        if math.isnan(msg.latitude) or msg.status.status < 0:
            return
        with self.lock:
            self.ekf = self.to_enu(msg.latitude, msg.longitude)

    def ekf_vel_cb(self, msg):
        with self.lock:
            self.ekf_vel = (msg.twist.linear.x, msg.twist.linear.y)

    def flow_cb(self, msg):
        with self.lock:
            self.flow_vel = (msg.twist.linear.x, msg.twist.linear.y)

    def diag_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.diag = d
            self.diag_stamps.append((d.get('stamp', 0.0),
                                     d.get('status') in ('MATCH', 'MATCH_INIT')))

    def map_cb(self, msg):
        with self.lock:
            self.map_frames = np.array([[p.position.x, p.position.y] for p in msg.poses])

    def gps_input_cb(self, _msg):
        self.gps_input_times.append(time.time())

    # --- commands
    def send_gimbal_cmd(self, roll, pitch, yaw):
        msg = MountControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.mode = 2
        msg.pitch, msg.roll, msg.yaw = float(pitch), float(roll), float(yaw)
        self.mount_pub.publish(msg)

    def set_mode(self, mode):
        self.mode_cli.call_async(SetMode.Request(custom_mode=mode))

    # --- snapshot for the GUI
    def snapshot(self):
        with self.lock:
            s = {k: getattr(self, k) for k in (
                'state', 'sim_now', 'gt', 'gt_vel', 'est', 'ekf', 'ekf_vel', 'ekf_alt',
                'flow_vel', 'map_frames')}
            s['diag'] = dict(self.diag)
            stamps = list(self.diag_stamps)
        s['heading'] = self.heading
        # Frame rate and fix rate over the last 5 s of simulated time.
        if stamps:
            t_end = stamps[-1][0]
            recent = [x for x in stamps if t_end - x[0] <= 5.0]
            span = recent[-1][0] - recent[0][0] if len(recent) > 1 else 0.0
            s['fps'] = (len(recent) - 1) / span if span > 0 else None
            s['fix_rate'] = 100.0 * sum(x[1] for x in recent) / len(recent)
        else:
            s['fps'] = s['fix_rate'] = None
        now = time.time()
        recent_gi = [t for t in self.gps_input_times if now - t <= 2.0]
        s['gps_input_hz'] = len(recent_gi) / 2.0
        return s


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
def fmt(value, spec='.2f', unit=''):
    return '—' if value is None else f'{value:{spec}}{unit}'


def norm(v):
    return None if v is None else math.hypot(*v)


class Card(QGroupBox):
    """A titled box of label/value rows."""

    def __init__(self, title, rows):
        super().__init__(title)
        self.setStyleSheet('QGroupBox { font-weight: bold; }')
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignLeft)
        self.values = {}
        for key, label in rows:
            v = QLabel('—')
            v.setStyleSheet('font-family: monospace; font-size: 13px;')
            v.setAlignment(Qt.AlignRight)
            form.addRow(QLabel(label), v)
            self.values[key] = v

    def set(self, key, text, color=None):
        lbl = self.values[key]
        lbl.setText(text)
        lbl.setStyleSheet('font-family: monospace; font-size: 13px;'
                          + (f' color: {color}; font-weight: bold;' if color else ''))


class TimePlot(pg.PlotWidget):
    """Rolling time-series plot with named curves."""

    def __init__(self, title, unit, curves):
        super().__init__()
        self.setTitle(title)
        self.setLabel('left', unit)
        self.setLabel('bottom', 'sim time', 's')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.addLegend(offset=(5, 5))
        self.curves = {name: self.plot([], [], name=name, pen=pg.mkPen(color, width=2))
                       for name, color in curves}
        self.data = {name: deque(maxlen=3000) for name, _ in curves}

    def append(self, t, values):
        for name, v in values.items():
            if v is not None:
                self.data[name].append((t, v))

    def redraw(self, window_s):
        for name, curve in self.curves.items():
            d = self.data[name]
            if not d:
                continue
            arr = np.array(d)
            arr = arr[arr[:, 0] >= arr[-1, 0] - window_s]
            curve.setData(arr[:, 0], arr[:, 1])


class MainWindow(QMainWindow):
    image_signal = Signal(np.ndarray)

    def __init__(self, node: GcsNode):
        super().__init__()
        self.node = node
        self.setWindowTitle('Visual Navigation GCS')
        self.resize(1500, 950)
        node.image_signal = self.image_signal
        self.image_signal.connect(self.update_cam_display)

        self.paths = {k: deque(maxlen=20000) for k in ('gt', 'est', 'ekf')}
        self.map_drawn = False
        self.last_sample_t = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self.build_status_bar())
        tabs = QTabWidget()
        tabs.addTab(self.build_nav_tab(), 'Navigation')
        tabs.addTab(self.build_camera_tab(), 'Camera && Gimbal')
        tabs.addTab(self.build_modes_tab(), 'Flight Modes')
        layout.addWidget(tabs)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    # --- layout
    def build_status_bar(self):
        frame = QFrame()
        frame.setStyleSheet('background-color: #333; color: white; border-radius: 5px;')
        row = QHBoxLayout(frame)
        font = QFont('Arial', 12, QFont.Bold)
        self.lbl_mode, self.lbl_arm, self.lbl_status = QLabel('MODE: —'), QLabel('ARM: —'), QLabel('')
        for lbl in (self.lbl_mode, self.lbl_status, self.lbl_arm):
            lbl.setFont(font)
        row.addWidget(self.lbl_mode)
        row.addStretch()
        row.addWidget(self.lbl_status)
        row.addStretch()
        row.addWidget(self.lbl_arm)
        return frame

    def build_nav_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)

        # Left: status cards
        left = QVBoxLayout()
        self.card_loc = Card('Localization', [
            ('status', 'Status'), ('since', 'Since last fix'), ('inliers', 'Inliers'),
            ('sigma', 'Sigma'), ('latency', 'Latency'), ('fps', 'Frame rate'),
            ('fixrate', 'Fix rate (5 s)'), ('cands', 'Candidates'), ('radius', 'Search radius')])
        self.card_err = Card('Error (estimate − truth)', [
            ('total', 'Total'), ('along', 'Along track'), ('cross', 'Cross track'),
            ('ekf', 'EKF − truth')])
        self.card_speed = Card('Speed', [
            ('gt', 'Truth'), ('ekf', 'EKF'), ('flow', 'Optical flow'), ('gi', 'GPS_INPUT rate')])
        self.card_alt = Card('Altitude', [('gt', 'Truth'), ('ekf', 'EKF (rel_alt)')])
        self.card_map = Card('Map', [
            ('nearest', 'Nearest map frame'), ('ref', 'Matched frame'), ('refdist', 'Fix offset'),
            ('target', 'To target'), ('progress', 'Progress')])
        for card in (self.card_loc, self.card_err, self.card_speed, self.card_alt, self.card_map):
            left.addWidget(card)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(330)
        outer.addWidget(left_w)

        # Right: map + time plots
        right = QVBoxLayout()
        controls = QHBoxLayout()
        # Follow keeps the vehicle centered at the current zoom level; dragging
        # the map turns it off, the mouse wheel zooms around the vehicle.
        self.chk_follow = QCheckBox('Follow vehicle')
        self.chk_follow.setChecked(True)
        btn_fit = QPushButton('Fit all')
        btn_fit.clicked.connect(self.fit_map)
        self.sp_window = QSpinBox()
        self.sp_window.setRange(10, 1800)
        self.sp_window.setValue(120)
        self.sp_window.setSuffix(' s')
        btn_clear = QPushButton('Clear paths')
        btn_clear.clicked.connect(self.clear_paths)
        controls.addWidget(self.chk_follow)
        controls.addWidget(btn_fit)
        controls.addWidget(QLabel('Plot window'))
        controls.addWidget(self.sp_window)
        controls.addWidget(btn_clear)
        controls.addStretch()
        right.addLayout(controls)

        self.map_plot = pg.PlotWidget(title='Map (ENU, m)')
        self.map_plot.setAspectLocked(True)
        self.map_plot.showGrid(x=True, y=True, alpha=0.3)
        self.map_plot.addLegend(offset=(5, 5))
        self.map_frames_item = pg.ScatterPlotItem(size=7, pen=pg.mkPen('#006064', width=1),
                                                  brush=pg.mkBrush(0, 229, 255, 170),
                                                  name='Map frames (survey images)')
        self.map_plot.addItem(self.map_frames_item)
        tx, ty = self.node.target
        self.map_plot.plot([0, tx], [0, ty], pen=pg.mkPen('#1e88e5', width=1, style=Qt.DashLine),
                           name='Planned path')
        self.map_plot.plot([tx], [ty], pen=None, symbol='x', symbolSize=14, symbolBrush='#1e88e5')
        self.path_items = {
            'gt': self.map_plot.plot([], [], pen=pg.mkPen('#43a047', width=2), name='Truth'),
            'ekf': self.map_plot.plot([], [], pen=pg.mkPen('#8e24aa', width=1), name='EKF'),
            'est': self.map_plot.plot([], [], pen=pg.mkPen('#e53935', width=2), name='Visual estimate'),
        }
        self.ref_item = pg.ScatterPlotItem(size=16, pen=pg.mkPen('#fdd835', width=3), brush=None,
                                           name='Matched frame')
        self.fix_item = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush('#ff7043'), name='Raw fix')
        self.radius_item = pg.PlotDataItem(pen=pg.mkPen('#fdd835', width=1, style=Qt.DotLine))
        self.vehicle_items = {
            'gt': pg.ScatterPlotItem(size=12, brush=pg.mkBrush('#43a047')),
            'est': pg.ScatterPlotItem(size=12, brush=pg.mkBrush('#e53935')),
        }
        for item in (self.ref_item, self.fix_item, self.radius_item, *self.vehicle_items.values()):
            self.map_plot.addItem(item)
        self.map_plot.setRange(xRange=(-120, 120), yRange=(-120, 120), padding=0)
        self.map_width = None
        self.map_plot.getViewBox().sigRangeChangedManually.connect(self.on_map_manual_range)
        right.addWidget(self.map_plot, stretch=3)

        grid = QGridLayout()
        self.plot_err = TimePlot('Position error', 'm', [
            ('total', '#e53935'), ('along', '#1e88e5'), ('cross', '#43a047')])
        self.plot_speed = TimePlot('Horizontal speed', 'm/s', [
            ('truth', '#43a047'), ('EKF', '#8e24aa'), ('flow', '#fb8c00')])
        self.plot_latency = TimePlot('Localization latency', 'ms', [('latency', '#e53935')])
        grid.addWidget(self.plot_err, 0, 0)
        grid.addWidget(self.plot_speed, 0, 1)
        grid.addWidget(self.plot_latency, 0, 2)
        grid_w = QWidget()
        grid_w.setLayout(grid)
        right.addWidget(grid_w, stretch=1)
        outer.addLayout(right, stretch=1)
        return tab

    def build_camera_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.lbl_video = QLabel('NO VIDEO')
        self.lbl_video.setMinimumSize(640, 480)
        self.lbl_video.setStyleSheet('background-color: black; color: white; border: 2px solid gray;')
        self.lbl_video.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_video, stretch=3)

        panel = QVBoxLayout()
        box = QGroupBox('Gimbal Control')
        form = QFormLayout(box)
        self.sp_roll = QDoubleSpinBox()
        self.sp_roll.setRange(-90, 90)
        self.sp_pitch = QDoubleSpinBox()
        self.sp_pitch.setRange(-90, 0)
        self.sp_pitch.setValue(-90)
        self.sp_yaw = QDoubleSpinBox()
        self.sp_yaw.setRange(-180, 180)
        form.addRow('Roll:', self.sp_roll)
        form.addRow('Pitch:', self.sp_pitch)
        form.addRow('Yaw:', self.sp_yaw)
        btn = QPushButton('Send Angle')
        btn.setMinimumHeight(40)
        btn.setStyleSheet('background-color: #007bff; color: white; font-weight: bold;')
        btn.clicked.connect(lambda: self.node.send_gimbal_cmd(
            self.sp_roll.value(), self.sp_pitch.value(), self.sp_yaw.value()))
        panel.addWidget(box)
        panel.addWidget(btn)
        panel.addStretch()
        w = QWidget()
        w.setLayout(panel)
        layout.addWidget(w, stretch=1)
        return tab

    def build_modes_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)
        modes = [('STABILIZE', 0, 0), ('LOITER', 0, 1), ('POSHOLD', 0, 2),
                 ('GUIDED', 1, 0), ('AUTO', 1, 1), ('RTL', 1, 2),
                 ('LAND', 2, 0), ('BRAKE', 2, 1), ('ALT_HOLD', 2, 2)]
        for name, r, c in modes:
            btn = QPushButton(name)
            btn.setMinimumHeight(80)
            color = {'LAND': '#d9534f', 'RTL': '#f0ad4e'}.get(name)
            btn.setStyleSheet(
                f'background-color: {color}; color: white; font-weight: bold; font-size: 16px; '
                'border-radius: 10px;' if color else
                'QPushButton { font-size: 16px; font-weight: bold; background-color: #e1e1e1; '
                'border-radius: 10px; } QPushButton:hover { background-color: #d4d4d4; }')
            btn.clicked.connect(lambda checked=False, n=name: self.node.set_mode(n))
            layout.addWidget(btn, r, c)
        return tab

    # --- updates
    def clear_paths(self):
        for p in self.paths.values():
            p.clear()

    def refresh(self):
        s = self.node.snapshot()
        d = s['diag']
        self.update_status(s, d)

        gt, est, ekf = s['gt'], s['est'], s['ekf']
        for key, pos in (('gt', gt), ('est', est), ('ekf', ekf)):
            if pos is not None:
                path = self.paths[key]
                if not path or math.hypot(pos[0] - path[-1][0], pos[1] - path[-1][1]) > 0.3:
                    path.append((pos[0], pos[1]))

        # Errors: along/cross relative to the true direction of travel
        # (or the planned path when hovering).
        err = along = cross = ekf_err = None
        if gt is not None and est is not None:
            ex, ey = est[0] - gt[0], est[1] - gt[1]
            err = math.hypot(ex, ey)
            v = s['gt_vel']
            if v is not None and norm(v) > 0.5:
                ux, uy = v[0] / norm(v), v[1] / norm(v)
            else:
                tx, ty = self.node.target
                ln = math.hypot(tx, ty) or 1.0
                ux, uy = tx / ln, ty / ln
            along = ex * ux + ey * uy
            cross = -ex * uy + ey * ux
        if gt is not None and ekf is not None:
            ekf_err = math.hypot(ekf[0] - gt[0], ekf[1] - gt[1])

        self.update_cards(s, d, err, along, cross, ekf_err)
        self.update_map(s, d)

        t = s['sim_now']
        if t is not None and t != self.last_sample_t:
            self.last_sample_t = t
            self.plot_err.append(t, {'total': err, 'along': along, 'cross': cross})
            self.plot_speed.append(t, {'truth': norm(s['gt_vel']), 'EKF': norm(s['ekf_vel']),
                                       'flow': norm(s['flow_vel'])})
            self.plot_latency.append(t, {'latency': d.get('ms_total')})
        window = self.sp_window.value()
        for plot in (self.plot_err, self.plot_speed, self.plot_latency):
            plot.redraw(window)

    def update_status(self, s, d):
        st = s['state']
        self.lbl_mode.setText(f'MODE: {st.mode or "—"}')
        self.lbl_arm.setText(f'ARM: {"YES" if st.armed else "NO"}')
        self.lbl_arm.setStyleSheet('color: #ff5252;' if st.armed else 'color: white;')
        status = d.get('status')
        if status:
            self.lbl_status.setText(f'{d.get("algorithm", "")} {status}')
            self.lbl_status.setStyleSheet(f'color: {STATUS_COLORS.get(status, "white")};')

    def update_cards(self, s, d, err, along, cross, ekf_err):
        status = d.get('status')
        self.card_loc.set('status', status or '—', STATUS_COLORS.get(status))
        self.card_loc.set('since', fmt(d.get('since_fix_s'), '.1f', ' s'))
        self.card_loc.set('inliers', fmt(d.get('inliers'), 'd'))
        self.card_loc.set('sigma', fmt(d.get('sigma_m'), '.2f', ' m'))
        self.card_loc.set('latency', fmt(d.get('ms_total'), '.0f', ' ms'))
        self.card_loc.set('fps', fmt(s['fps'], '.1f', ' Hz'))
        self.card_loc.set('fixrate', fmt(s['fix_rate'], '.0f', ' %'))
        self.card_loc.set('cands', fmt(d.get('n_candidates'), 'd'))
        self.card_loc.set('radius', fmt(d.get('search_radius_m'), '.0f', ' m'))

        self.card_err.set('total', fmt(err, '.2f', ' m'))
        self.card_err.set('along', fmt(along, '+.2f', ' m'))
        self.card_err.set('cross', fmt(cross, '+.2f', ' m'))
        self.card_err.set('ekf', fmt(ekf_err, '.2f', ' m'))

        self.card_speed.set('gt', fmt(norm(s['gt_vel']), '.2f', ' m/s'))
        self.card_speed.set('ekf', fmt(norm(s['ekf_vel']), '.2f', ' m/s'))
        self.card_speed.set('flow', fmt(norm(s['flow_vel']), '.2f', ' m/s'))
        hz = s['gps_input_hz']
        self.card_speed.set('gi', f'{hz:.1f} Hz', '#c62828' if hz < 4.0 else None)

        gt = s['gt']
        self.card_alt.set('gt', fmt(gt[2] if gt else None, '.1f', ' m'))
        self.card_alt.set('ekf', fmt(s['ekf_alt'], '.1f', ' m'))

        self.card_map.set('nearest', fmt(d.get('nearest_map_frame_m'), '.1f', ' m'))
        ref = d.get('ref_file')
        self.card_map.set('ref', ref.rsplit('_', 1)[-1] if ref else '—')
        if 'fix_x' in d:
            self.card_map.set('refdist', fmt(math.hypot(d['fix_x'] - d['ref_x'],
                                                        d['fix_y'] - d['ref_y']), '.1f', ' m'))
        pos = s['est'] or gt
        if pos is not None:
            tx, ty = self.node.target
            total = math.hypot(tx, ty) or 1.0
            self.card_map.set('target', f'{math.hypot(tx - pos[0], ty - pos[1]):.0f} m')
            prog = (pos[0] * tx + pos[1] * ty) / total / total * 100.0
            self.card_map.set('progress', f'{min(max(prog, 0.0), 100.0):.0f} %')

    def update_map(self, s, d):
        if not self.map_drawn and s['map_frames'] is not None and len(s['map_frames']):
            mf = s['map_frames']
            self.map_frames_item.setData(mf[:, 0], mf[:, 1])
            self.map_drawn = True
        for key, item in self.path_items.items():
            p = self.paths[key]
            if p:
                arr = np.array(p)
                item.setData(arr[:, 0], arr[:, 1])
        for key, item in self.vehicle_items.items():
            pos = s[key]
            item.setData([pos[0]], [pos[1]]) if pos is not None else item.clear()
        if 'ref_x' in d:
            self.ref_item.setData([d['ref_x']], [d['ref_y']])
            self.fix_item.setData([d['fix_x']], [d['fix_y']])
        r = d.get('search_radius_m')
        if r and d.get('x') is not None:
            a = np.linspace(0, 2 * np.pi, 60)
            self.radius_item.setData(d['x'] + r * np.cos(a), d['y'] + r * np.sin(a))
        else:
            self.radius_item.clear()

        center = s['gt'] or s['est']
        if self.chk_follow.isChecked() and center is not None:
            # Re-center on the vehicle but keep whatever zoom the user chose.
            (x0, x1), (y0, y1) = self.map_plot.getViewBox().viewRange()
            hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            self.map_plot.setRange(xRange=(center[0] - hx, center[0] + hx),
                                   yRange=(center[1] - hy, center[1] + hy), padding=0)
            self.map_width = 2.0 * hx

    def on_map_manual_range(self, *_):
        """Mouse interaction on the map: a drag (same width) stops following, a zoom does not."""
        (x0, x1), _ = self.map_plot.getViewBox().viewRange()
        width = x1 - x0
        if self.map_width is not None and abs(width - self.map_width) <= 1e-6 * max(width, 1.0):
            self.chk_follow.setChecked(False)
        self.map_width = width

    def fit_map(self):
        """Show the whole map and every path; stops following."""
        self.chk_follow.setChecked(False)
        pts = [np.array(p) for p in self.paths.values() if p]
        mf = self.node.snapshot()['map_frames']
        if mf is not None and len(mf):
            pts.append(mf)
        pts.append(np.array([[0.0, 0.0], self.node.target]))
        allp = np.vstack(pts)
        self.map_plot.setRange(xRange=(allp[:, 0].min(), allp[:, 0].max()),
                               yRange=(allp[:, 1].min(), allp[:, 1].max()), padding=0.05)
        self.map_width = None

    @Slot(np.ndarray)
    def update_cam_display(self, cv_img):
        h, w, ch = cv_img.shape
        rgb = np.ascontiguousarray(cv_img[..., ::-1])
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))


def main():
    rclpy.init()
    node = GcsNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow(node)
    win.show()
    try:
        sys.exit(app.exec())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

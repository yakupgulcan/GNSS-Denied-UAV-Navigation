#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from gnss_denied_nav.orb_detect_match import ORBDetectAndMatch

class VisualBacktrackTester(Node):
    def __init__(self):
        super().__init__('visual_backtrack_tester')

        # --- AYARLAR ---
        self.declare_parameter('db_path', '')
        self.DB_PATH = self.get_parameter('db_path').get_parameter_value().string_value
        self.TARGET_ALT = 29.0   
        self.RECORD_DURATION = 30.0 
        self.RECORD_SPEED = 2.0     
        self.BACKTRACK_SPEED = -1.0 
        self.KEYFRAME_INTERVAL = 1.0 
        
        self.MIN_TIME_PER_WAYPOINT = 2.0 
        self.last_waypoint_switch_time = 0.0

        # --- DURUM ---
        self.mission_state = "TAKEOFF" # Başlangıç durumu
        self.start_time = 0
        self.last_keyframe_time = 0
        self.current_alt = 0.0
        self.current_heading = 0.0
        self.mavros_state = State()

        self.breadcrumb_stack = deque()

        # İletişim
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos)
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(Image, '/camera/image', self.image_callback, qos)

        self.pub_speed = self.create_publisher(Float64, '/control/target_speed', 10)
        self.pub_yaw = self.create_publisher(Float64, '/control/target_yaw', 10)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.pub_status = self.create_publisher(String, '/test/status', 10)

        self.client_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.client_mode = self.create_client(SetMode, '/mavros/set_mode')

        self.bridge = CvBridge()
        self.localizer = ORBDetectAndMatch(self.DB_PATH, n_features=1000)
        
        self.create_timer(0.05, self.control_loop)

    def alt_cb(self, msg): self.current_alt = msg.data
    def hdg_cb(self, msg): self.current_heading = msg.data
    def state_cb(self, msg): self.mavros_state = msg

    # --- ANA KONTROL DÖNGÜSÜ ---
    def control_loop(self):
        self.pub_status.publish(String(data=self.mission_state))

        # 1. KALKIŞ
        if self.mission_state == "TAKEOFF":
            if not self.mavros_state.connected: return
            if self.mavros_state.mode != "STABILIZE":
                self.set_mode("STABILIZE")
                self.send_direct_rc(1500, 1500, 1500, 1500)
                return
            if not self.mavros_state.armed:
                self.arm_vehicle()
                self.send_direct_rc(1500, 1500, 1000, 1500)
                return

            error = self.TARGET_ALT - self.current_alt
            if abs(error) < 1.0:
                self.get_logger().info("Kalkış Tamam. ALT_HOLD ve KAYIT başlıyor.")
                self.set_mode("ALT_HOLD")
                self.send_direct_rc(1500, 1500, 1500, 1500)
                self.mission_state = "RECORDING"
                self.start_time = time.time()
                return

            tgt_pwm = int(1500 + (error * 12.0)) # P-Kontrol
            tgt_pwm = max(1250, min(1750, tgt_pwm))
            self.send_direct_rc(1500, 1500, tgt_pwm, 1500)

        # 2. KAYIT (RECORDING)
        elif self.mission_state == "RECORDING":
            elapsed = time.time() - self.start_time
            if elapsed > self.RECORD_DURATION:
                self.get_logger().info("Süre Doldu. FRENLEME.")
                self.mission_state = "BRAKING"
                self.start_time = time.time()
                return
            
            # --- DÜZELTME: Yaw'ı Sabit 0.0'a Kilitle ---
            # Base Controller bunu alıp Kuzey'e (0 derece) kilitleyecek.
            self.send_topic_command(self.RECORD_SPEED, 0.0)

        # 3. FRENLEME
        elif self.mission_state == "BRAKING":
            self.send_topic_command(0.0, 0.0) # Hız 0, Yaw 0
            
            if (time.time() - self.start_time) > 4.0:
                self.get_logger().info("Frenleme Bitti. Stack Temizleniyor...")
                for _ in range(min(5, len(self.breadcrumb_stack))):
                    self.breadcrumb_stack.pop()
                
                self.get_logger().info(f"Geri Dönüş Başlıyor! Hedef: {len(self.breadcrumb_stack)} Kare")
                self.mission_state = "BACKTRACKING"
                self.last_waypoint_switch_time = time.time()

        # 4. BACKTRACKING
        elif self.mission_state == "BACKTRACKING":
            if len(self.breadcrumb_stack) == 0:
                self.get_logger().info("Tüm izler bitti! İNİŞ.")
                self.mission_state = "LANDING"
                self.set_mode("LAND")
                return

    # --- GÖRÜNTÜ İŞLEME ---
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            
            if self.mission_state == "RECORDING":
                now = time.time()
                if (now - self.last_keyframe_time) > self.KEYFRAME_INTERVAL:
                    kp, des = self.localizer.orb.detectAndCompute(cv_image, None)
                    if des is not None:
                        # Hdg kaydetmeye gerek yok, zaten hep 0.0 gittik.
                        # Ama dönüşte referans olsun diye kaydediyoruz.
                        self.breadcrumb_stack.append({'des': des, 'hdg': self.current_heading})
                        self.last_keyframe_time = now
                        self.get_logger().info(f"Kayıt: {len(self.breadcrumb_stack)}. Kare")

            elif self.mission_state == "BACKTRACKING":
                if not self.breadcrumb_stack: return

                target_data = self.breadcrumb_stack[-1]
                target_des = target_data['des']
                target_hdg = target_data['hdg']
                
                # --- BURASI ARTIK 6 DEĞER DÖNDÜRÜYOR ---
                res = self.localizer.match_frame_to_descriptors(cv_image, target_des, min_match_count=6)
                
                # Varsayılanlar
                target_yaw = target_hdg
                speed_cmd = self.BACKTRACK_SPEED

                if res:
                    # Unpack 6 Values (Hata burada çözüldü)
                    matches, err_x, err_y, spread, cx, cy = res
                    
                    # Yaw Correction
                    yaw_correction = err_x * 0.05 
                    target_yaw = target_hdg + yaw_correction
                    
                    now = time.time()
                    dt = now - self.last_waypoint_switch_time
                    
                    if matches > 35 and dt > self.MIN_TIME_PER_WAYPOINT:
                        self.get_logger().info(f" >>> GEÇİLDİ (M:{matches}). Kalan: {len(self.breadcrumb_stack)}")
                        self.breadcrumb_stack.pop()
                        self.last_waypoint_switch_time = now
                    
                    self.send_topic_command(speed_cmd, target_yaw)
                
                else:
                    self.get_logger().warn("İz Kaybedildi! Referans Yaw'a dönülüyor...")
                    self.send_topic_command(0.0, target_hdg)

        except Exception as e:
            self.get_logger().error(f"Img Error: {e}")

    # --- YARDIMCI KOMUTLAR ---
    def send_topic_command(self, speed, yaw):
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_yaw.publish(Float64(data=float(yaw)))

    def send_direct_rc(self, roll, pitch, throttle, yaw):
        msg = OverrideRCIn()
        msg.channels = [65535]*18
        msg.channels[0]=int(roll); msg.channels[1]=int(pitch); 
        msg.channels[2]=int(throttle); msg.channels[3]=int(yaw)
        self.rc_pub.publish(msg)

    def set_mode(self, mode):
        if self.mavros_state.mode != mode:
            self.client_mode.call_async(SetMode.Request(custom_mode=mode))

    def arm_vehicle(self):
        if not self.mavros_state.armed:
            self.client_arm.call_async(CommandBool.Request(value=True))

def main():
    rclpy.init()
    node = VisualBacktrackTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_direct_rc(1500, 1500, 1500, 1500)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
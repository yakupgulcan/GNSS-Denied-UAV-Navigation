#!/usr/bin/env python3
"""
v11_akaze: Fixed Initialization & Robust Matching
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, NavSatFix
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64, String, Bool
from mavros_msgs.msg import State
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
import csv
from datetime import datetime
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- IMPORTLAR ---
from gnss_denied_nav.akaze_detect_match import AKAZEDetectAndMatch

MATCH_ALGO = 4 
DB_PATH = ""
START_LAT = -35.3658674
START_LON = 149.1652376

class VisualLocalizationNode(Node):
    def __init__(self):
        super().__init__('visual_localization_node_akaze')
        self.is_landing = False
        self.parallel_cb_group = ReentrantCallbackGroup()

        # CSV Log
        ct = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f'visual_log_akaze_{ct}.csv'
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'source', 'x', 'y', 'state'])
        
        # Parametreler
        self.declare_parameter('db_path', DB_PATH)
        actual_db_path = self.get_parameter('db_path').get_parameter_value().string_value
        
        self.get_logger().info('AKAZE Veritabanı yükleniyor...')
        # Threshold 0.001 gürültüyü azaltır
        self.localizer = AKAZEDetectAndMatch(actual_db_path, threshold=0.001)
        self.get_logger().info('AKAZE Veritabanı yüklendi.')

        # --- DURUM DEĞİŞKENLERİ ---
        self.state = "NORMAL" 
        self.mavros_state = State()
        
        # Konum Başlatma Bayrağı (ÖNEMLİ)
        self.is_localized = False 
        
        self.current_x = 0.0 
        self.current_y = 0.0 
        self.current_vel_x = 0.0
        self.current_vel_y = 0.0
        self.current_heading = 0.0
        self.last_vel_time = time.time()
        
        # --- BREADCRUMB AYARLARI ---
        self.last_db_success_time = time.time() 
        self.DB_TIMEOUT = 10.0 # 10 sn veri gelmezse Backtrack
        
        self.KEYFRAME_INTERVAL = 0.5 # Daha sık kayıt al (İz takibi için)
        self.last_keyframe_time = 0
        self.breadcrumb_stack = deque(maxlen=3000) 
        
        # Backtrack Mantığı
        self.MIN_TIME_PER_WAYPOINT = 0.5
        self.last_waypoint_switch_time = 0.0
        self.BACKTRACK_SPEED = -1.0
        
        self.hover_start_time = 0.0
        self.HOVER_DURATION = 3.0 
        self.brake_start_time = 0.0
        self.BRAKE_DURATION = 2.0 
        self.BACKTRACK_SKIP_COUNT = 5 

        # --- YAYINCILAR & ABONELİKLER ---
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose_enu', 10)
        self.gps_publisher = self.create_publisher(NavSatFix, '/visual_gps', qos_sensor)
        self.pub_status = self.create_publisher(String, '/system/status', 10)
        self.pub_override = self.create_publisher(Bool, '/control/override', 10)
        self.pub_speed = self.create_publisher(Float64, '/control/target_speed', 10)
        self.pub_yaw = self.create_publisher(Float64, '/control/target_yaw', 10)

        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.vel_sub = self.create_subscription(TwistStamped, '/visual_velocity_enu', self.velocity_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.hdg_sub = self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.alt_sub = self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos_sensor, callback_group=self.parallel_cb_group)
        
        self.bridge = CvBridge()
        self.SMOOTHING_ALPHA = 0.3 # Daha yumuşak geçiş
        self.current_altitude = 0.0

    def alt_cb(self, msg): self.current_altitude = msg.data
    def hdg_cb(self, msg): self.current_heading = msg.data
    def state_cb(self, msg): self.mavros_state = msg

    # --- HIZ CALLBACK ---
    def velocity_callback(self, msg):
        if self.mavros_state.mode == "LAND": return
        now = time.time()
        dt = now - self.last_vel_time
        self.last_vel_time = now
        if dt > 1.0: dt = 0.0 

        self.current_vel_x = msg.twist.linear.x
        self.current_vel_y = msg.twist.linear.y
        self.current_x += msg.twist.linear.x * dt
        self.current_y += msg.twist.linear.y * dt
        self.publish_current_pose() 

    # --- GÖRÜNTÜ CALLBACK ---
    def image_callback(self, msg):
        if self.mavros_state.mode == "LAND" or self.is_landing: 
            if self.mavros_state.mode == "LAND": self.is_landing = True
            return
        
        # 3 metreden aşağıda visual processing yapma (Yeri öpmesin)
        if self.current_altitude < 3.0: return 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            # --- ÖN İŞLEME: CLAHE ---
            # Kontrastı artırarak AKAZE'nin daha iyi çalışmasını sağlar
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cv_image = clahe.apply(cv_image)
            
            now = time.time()
            
            # --- KAYIT (BREADCRUMB) ---
            if self.state == "NORMAL":
                if (now - self.last_keyframe_time) > self.KEYFRAME_INTERVAL:
                    kp, des = self.localizer.akaze.detectAndCompute(cv_image, None)
                    if des is not None and len(des) > 5:
                        self.breadcrumb_stack.append({'des': des, 'hdg': self.current_heading})
                        self.last_keyframe_time = now

            # --- 1. DB EŞLEŞMESİ ARA ---
            db_match_pos = self.try_get_db_position(cv_image)
            
            if db_match_pos is not None:
                match_x, match_y = db_match_pos
                
                # --- OUTLIER KONTROLÜ (GÜNCELLENDİ) ---
                # Eğer daha önce hiç konum bulamadıysak (is_localized=False),
                # ilk bulduğumuz konumu kabul ediyoruz (INIT).
                if not self.is_localized:
                    self.get_logger().warn(f"İLK KİLİTLEME (INIT): X={match_x:.1f}, Y={match_y:.1f}")
                    self.current_x = match_x
                    self.current_y = match_y
                    self.is_localized = True
                    self.last_db_success_time = now
                    return

                # Normal Süreç:
                elapsed_time = now - self.last_db_success_time
                current_speed = math.sqrt(self.current_vel_x**2 + self.current_vel_y**2)
                
                # Toleransı artırdık: (Hız * Zaman * 5) + 25 metre
                # 0.8m hatayı reddetmemesi için sabit değer 25.0 yapıldı.
                acceptable_diff = (current_speed * elapsed_time * 5.0) + 25.0 
                
                dist_error = math.sqrt((match_x - self.current_x)**2 + (match_y - self.current_y)**2)
                
                if dist_error < acceptable_diff:
                    # GEÇERLİ MATCH
                    self.last_db_success_time = now
                    self.current_x = self.SMOOTHING_ALPHA * match_x + (1 - self.SMOOTHING_ALPHA) * self.current_x
                    self.current_y = self.SMOOTHING_ALPHA * match_y + (1 - self.SMOOTHING_ALPHA) * self.current_y
                    
                    if self.state in ["BACKTRACKING", "BRAKING", "HOVER_WAIT"]:
                        self.get_logger().info("AKAZE EŞLEŞTİ! NORMAL MOD.")
                        self.switch_to_normal_mode()

                    self.log_to_csv("DB_MATCH", self.current_x, self.current_y)
                else:
                    self.get_logger().warn(f"Outlier Reddedildi: Fark {dist_error:.1f}m > Limit {acceptable_diff:.1f}m")

            # --- [B] DB EŞLEŞMESİ YOK ---
            else:
                time_since_last_match = now - self.last_db_success_time
                
                if self.state == "NORMAL":
                    if time_since_last_match > self.DB_TIMEOUT:
                        self.get_logger().error(f"TIMEOUT ({self.DB_TIMEOUT}s)! BACKTRACK BAŞLIYOR.")
                        self.switch_to_backtrack_mode()
                
                elif self.state == "BRAKING":
                    self.send_control(0.0, self.current_heading)
                    if (now - self.brake_start_time) > self.BRAKE_DURATION:
                        self.get_logger().info("Fren bitti. Geri dönüş.")
                        self.state = "BACKTRACKING"
                        for _ in range(min(self.BACKTRACK_SKIP_COUNT, len(self.breadcrumb_stack))):
                            self.breadcrumb_stack.pop()
                        self.last_waypoint_switch_time = now

                elif self.state == "BACKTRACKING":
                    self.process_backtrack_logic(cv_image)
                
                elif self.state == "HOVER_WAIT":
                    self.send_control(0.0, self.current_heading) 
                    if (now - self.hover_start_time) > self.HOVER_DURATION:
                        self.get_logger().info("Bekleme bitti. Devam...")
                        self.state = "BACKTRACKING"
                        if self.breadcrumb_stack: self.breadcrumb_stack.pop()
            
            self.pub_status.publish(String(data=self.state))

        except Exception as e:
            self.get_logger().error(f'Hata: {str(e)}')

    # --- YARDIMCI FONKSİYONLAR ---
    def switch_to_backtrack_mode(self):
        self.state = "BRAKING"
        self.brake_start_time = time.time()
        self.pub_override.publish(Bool(data=True)) 
        
    def switch_to_normal_mode(self):
        self.state = "NORMAL"
        self.send_control(0.0, self.current_heading)
        self.pub_override.publish(Bool(data=False))

    def try_get_db_position(self, cv_image):
        # Min match count 8'den 6'ya düşürüldü (Daha toleranslı)
        candidates = self.localizer.get_location(cv_image, min_match_count=6, top_k=5)
        if not candidates: return None

        valid_points = []
        for gps_data, fname, score in candidates:
            if not isinstance(gps_data, dict): continue
            raw_lat = gps_data.get('lat') or gps_data.get('latitude')
            raw_lon = gps_data.get('lon') or gps_data.get('longitude')
            if raw_lat is None: continue
            
            x, y = self.latlon_to_enu(float(raw_lat), float(raw_lon))
            valid_points.append({'x': x, 'y': y, 'score': score})

        if not valid_points: return None

        # En iyi skora sahip olanı al, diğerleriyle ortalama yapma (AKAZE için daha güvenli)
        # Veya ağırlıklı ortalama devam edebilir
        best = valid_points[0]
        # Mekansal tutarlılık (50 metre içindekileri al)
        filtered = [p for p in valid_points if math.sqrt((p['x']-best['x'])**2 + (p['y']-best['y'])**2) < 50.0]
        
        if not filtered: return None

        sx=0; sy=0; sw=0
        for p in filtered:
            w = p['score']**2
            sx += p['x']*w; sy += p['y']*w; sw += w
        
        if sw == 0: return None
        return (sx/sw, sy/sw)

    def process_backtrack_logic(self, cv_image):
        if not self.breadcrumb_stack:
            self.get_logger().info("İz bitti. Bekliyoruz...")
            self.switch_to_normal_mode()
            self.last_db_success_time = time.time()
            return

        target_data = self.breadcrumb_stack[-1]
        target_des = target_data['des']
        target_hdg = target_data['hdg']
        
        # AKAZE eşleştirme
        result = self.localizer.match_frame_to_descriptors(cv_image, target_des, min_match_count=5)
        
        if result:
            match_count, error_x, _, _, _, _ = result
            
            yaw_correction = error_x * 0.05 
            target_yaw = target_hdg + yaw_correction
            
            speed_cmd = self.BACKTRACK_SPEED
            
            now = time.time()
            dt = now - self.last_waypoint_switch_time
            
            # Eşleşme varsa bekle
            if match_count > 30 and dt > self.MIN_TIME_PER_WAYPOINT:
                self.get_logger().info(f"Waypoint (Skor: {match_count}). Bekleniyor...")
                self.state = "HOVER_WAIT"
                self.hover_start_time = now
                self.send_control(0.0, self.current_heading)
                return 
            
            self.send_control(speed_cmd, target_yaw)
        else:
            self.get_logger().warn("İz Kayıp! Dönülüyor...")
            self.send_control(0.0, target_hdg)

    # ... (Diğer fonksiyonlar aynı) ...
    def publish_current_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.current_x
        pose_msg.pose.position.y = self.current_y
        self.pose_pub.publish(pose_msg)
        final_lat, final_lon = self.enu_to_latlon(self.current_x, self.current_y)
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = "map"
        gps_msg.latitude = final_lat
        gps_msg.longitude = final_lon
        self.gps_publisher.publish(gps_msg)

    def send_control(self, speed, yaw):
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_yaw.publish(Float64(data=float(yaw)))

    def log_to_csv(self, source, x, y):
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.csv_writer.writerow([timestamp, source, x, y, self.state])
        self.csv_file.flush()

    def latlon_to_enu(self, lat, lon):
        d_lat = lat - START_LAT
        d_lon = lon - START_LON
        north = d_lat * 111132.0
        east = d_lon * (111132.0 * math.cos(math.radians(START_LAT)))
        return east, north

    def enu_to_latlon(self, x, y):
        d_lat = y / 111132.0
        d_lon = x / (111132.0 * math.cos(math.radians(START_LAT)))
        return START_LAT + d_lat, START_LON + d_lon

    def destroy_node(self):
        if hasattr(self, 'csv_file'): self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualLocalizationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin() 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
v10: Full-Flight Breadcrumbing + Stop-and-Wait Recovery Logic
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
from gnss_denied_nav.orb_detect_match import ORBDetectAndMatch

MATCH_ALGO = 0 
DB_PATH = ""
START_LAT = -35.3658674
START_LON = 149.1652376

class VisualLocalizationNode(Node):
    def __init__(self):
        super().__init__('visual_localization_node')
        self.is_landing = True
        self.parallel_cb_group = ReentrantCallbackGroup()

        # CSV Log
        ct = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f'visual_log_{ct}.csv'
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'source', 'x', 'y', 'state'])
        
        # Parametreler
        self.declare_parameter('db_path', DB_PATH)
        actual_db_path = self.get_parameter('db_path').get_parameter_value().string_value
        
        self.get_logger().info('Veritabanı yükleniyor...')
        self.localizer = ORBDetectAndMatch(actual_db_path, n_features=1000)
        self.get_logger().info('Veritabanı yüklendi.')

        # --- DURUM DEĞİŞKENLERİ ---
        self.state = "NORMAL" # NORMAL, BRAKING, BACKTRACKING, HOVER_WAIT
        self.mavros_state = State()
        
        self.current_x = 0.0 
        self.current_y = 0.0 
        self.current_vel_x = 0.0
        self.current_vel_y = 0.0
        self.current_heading = 0.0
        self.last_vel_time = time.time()
        
        # --- BREADCRUMB AYARLARI ---
        self.last_db_success_time = time.time() 
        self.DB_TIMEOUT = 12.0 
        
        self.KEYFRAME_INTERVAL = 1.0 
        self.last_keyframe_time = 0
        self.breadcrumb_stack = deque(maxlen=2000) # Tüm uçuşu sakla
        
        # Backtrack Mantığı
        self.MIN_TIME_PER_WAYPOINT = 0.1 
        self.last_waypoint_switch_time = 0.0
        self.BACKTRACK_SPEED = -1.0
        
        # Stop-and-Wait
        self.hover_start_time = 0.0
        self.HOVER_DURATION = 3.0 # Waypoint'e varınca 3 sn bekle
        
        # Momentum
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

        # Paralel Abonelikler
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.vel_sub = self.create_subscription(TwistStamped, '/visual_velocity_enu', self.velocity_callback, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.hdg_sub = self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.hdg_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile=qos_sensor, callback_group=self.parallel_cb_group)
        
        self.bridge = CvBridge()
        self.SMOOTHING_ALPHA = 0.4 

    def hdg_cb(self, msg): self.current_heading = msg.data
    def state_cb(self, msg): self.mavros_state = msg

    # --- HIZ CALLBACK ---
    def velocity_callback(self, msg):
        # LAND modundaysak tahmin yapmayı bırak (Drift olmasın)
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
        # İnişteysek işlem yapma
        #if self.mavros_state.mode == "LAND" or self.is_landing or  self.mavros_state.mode != "ALT_HOLD": 
        if self.mavros_state.mode == "LAND": 
            if self.mavros_state.mode == "LAND" or self.is_landing:
                self.is_landing = True
            return
        self.get_logger().info("Eslesme deneniyor.")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            now = time.time()
            
            # --- HER ZAMAN KAYIT AL (DAİMİ BREADCRUMB) ---
            # DB eşleşmesi olsa da olmasa da, ilerledikçe iz bırakıyoruz.
            # Ancak Backtrack modundayken kendi izimizi kaydetmemeliyiz (Loop oluşur).
            if self.state == "NORMAL":
                if (now - self.last_keyframe_time) > self.KEYFRAME_INTERVAL:
                    kp, des = self.localizer.orb.detectAndCompute(cv_image, None)
                    if des is not None:
                        # Stack'e ekle
                        self.breadcrumb_stack.append({'des': des, 'hdg': self.current_heading})
                        self.last_keyframe_time = now
                        # Stack çok büyürse baştan sil (FIFO değil, LIFO kullanacağız ama boyutu korumak lazım)
                        # deque maxlen bunu otomatik yapar.

            # --- 1. DB EŞLEŞMESİ ARA ---
            db_match_pos = self.try_get_db_position(cv_image)
            
            if db_match_pos is not None:
                # --- [A] SAĞLAM ROTA BULUNDU ---
                match_x, match_y = db_match_pos
                
                # Outlier Kontrolü
                elapsed_time = now - self.last_db_success_time
                current_speed = math.sqrt(self.current_vel_x**2 + self.current_vel_y**2)
                acceptable_diff = (current_speed * elapsed_time * 5.0) + 15.0 
                dist_error = math.sqrt((match_x - self.current_x)**2 + (match_y - self.current_y)**2)
                
                if dist_error < acceptable_diff:
                    # GEÇERLİ MATCH -> RESETLE
                    self.last_db_success_time = now
                    self.current_x = self.SMOOTHING_ALPHA * match_x + (1 - self.SMOOTHING_ALPHA) * self.current_x
                    self.current_y = self.SMOOTHING_ALPHA * match_y + (1 - self.SMOOTHING_ALPHA) * self.current_y
                    
                    # Eğer Backtrack veya Bekleme modundaysak -> NORMALE DÖN
                    if self.state in ["BACKTRACKING", "BRAKING", "HOVER_WAIT"]:
                        self.get_logger().info("DB EŞLEŞMESİ GELDİ! NORMAL MODA DÖNÜLÜYOR.")
                        self.switch_to_normal_mode()
                        
                        # Stack'i silmek yerine, bulunduğumuz yere kadar olan kısmı budayabiliriz.
                        # Ama güvenli olanı: Eğer DB bulduysak, artık o noktadan sonrasını tekrar kaydedebiliriz.
                        # İsteğe bağlı: self.breadcrumb_stack.clear() (Eski rotayı unutmak istersen)
                        # Şimdilik tutuyoruz, belki ilerde yine lazım olur.

                    self.log_to_csv("DB_MATCH", self.current_x, self.current_y)
                else:
                    self.get_logger().warn(f"Outlier Match Reddedildi: Fark {dist_error:.1f}m")

            # --- [B] DB EŞLEŞMESİ YOK ---
            else:
                time_since_last_match = now - self.last_db_success_time
                
                if self.state == "NORMAL":
                    # 10 saniye limitini aştı mı?
                    if time_since_last_match > self.DB_TIMEOUT:
                        self.get_logger().error(f"{self.DB_TIMEOUT} sn veri yok! BACKTRACK BAŞLATILIYOR.")
                        self.switch_to_backtrack_mode()
                
                elif self.state == "BRAKING":
                    self.send_control(0.0, self.current_heading)
                    if (now - self.brake_start_time) > self.BRAKE_DURATION:
                        self.get_logger().info("Fren bitti. Geri dönüş başlıyor.")
                        self.state = "BACKTRACKING"
                        # Son birkaç kareyi atla
                        for _ in range(min(self.BACKTRACK_SKIP_COUNT, len(self.breadcrumb_stack))):
                            self.breadcrumb_stack.pop()
                        self.last_waypoint_switch_time = now

                elif self.state == "BACKTRACKING":
                    self.process_backtrack_logic(cv_image)
                
                elif self.state == "HOVER_WAIT":
                    # Bekleme Süresi Kontrolü
                    self.send_control(0.0, self.current_heading) # Dur ve Bekle
                    if (now - self.hover_start_time) > self.HOVER_DURATION:
                        self.get_logger().info("Bekleme süresi doldu. DB gelmedi. Geriye devam...")
                        self.state = "BACKTRACKING"
                        # O anki waypoint'i tüketip devam et
                        if self.breadcrumb_stack: self.breadcrumb_stack.pop()
            
            self.pub_status.publish(String(data=self.state))

        except Exception as e:
            self.get_logger().error(f'Hata: {str(e)}')

    # --- MOD GEÇİŞLERİ ---
    def switch_to_backtrack_mode(self):
        self.state = "BRAKING"
        self.brake_start_time = time.time()
        self.pub_override.publish(Bool(data=True)) # Planner Sus
        
    def switch_to_normal_mode(self):
        self.state = "NORMAL"
        self.send_control(0.0, self.current_heading)
        self.pub_override.publish(Bool(data=False)) # Planner Konuş

    # --- DB MATCHING (AYNI) ---
    def try_get_db_position(self, cv_image):
        candidates = self.localizer.get_location(cv_image, min_match_count=8, top_k=5)
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
        best = valid_points[0]
        filtered = [p for p in valid_points if math.sqrt((p['x']-best['x'])**2 + (p['y']-best['y'])**2) < 50.0]
        if not filtered: return None
        sx=0; sy=0; sw=0
        for p in filtered:
            w = p['score']**2
            sx += p['x']*w; sy += p['y']*w; sw += w
        if sw == 0: return None
        return (sx/sw, sy/sw)

    # --- BACKTRACK MANTIĞI (GÜNCELLENDİ) ---
    def process_backtrack_logic(self, cv_image):
        
        if not self.breadcrumb_stack:
            self.get_logger().info("İz bitti. Bekliyoruz...")
            self.switch_to_normal_mode()
            self.last_db_success_time = time.time()
            return

        target_data = self.breadcrumb_stack[-1]
        target_des = target_data['des']
        target_hdg = target_data['hdg']
        
        result = self.localizer.match_frame_to_descriptors(cv_image, target_des, min_match_count=6)
        
        if result:
            match_count, error_x, _, _, _, _ = result
            
            yaw_correction = error_x * 0.05 
            target_yaw = target_hdg + yaw_correction
            
            speed_cmd = self.BACKTRACK_SPEED
            
            now = time.time()
            dt = now - self.last_waypoint_switch_time
            
            # --- DUR VE BEKLE MANTIĞI ---
            if match_count > 35 and dt > self.MIN_TIME_PER_WAYPOINT:
                self.get_logger().info(f"Waypoint'e varıldı (Skor: {match_count}). Durup DB bekleniyor...")
                self.state = "HOVER_WAIT"
                self.hover_start_time = now
                self.send_control(0.0, self.current_heading) # Dur
                return # Döngüden çık, bir sonraki tur HOVER_WAIT'e girecek
            
            self.send_control(speed_cmd, target_yaw)
        else:
            self.get_logger().warn("İz Kaybedildi! Referans açıya dönülüyor...")
            self.send_control(0.0, target_hdg)

    # ... (Diğer yardımcı fonksiyonlar aynı) ...
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
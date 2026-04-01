#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OpticalFlowEstimator:
    def __init__(self, focal_length_px=370.0):
        self.focal_length = focal_length_px
        
        # Lucas-Kanade: Pencere boyutunu büyüttük, gürültü azalır
        self.lk_params = dict(winSize=(31, 31),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.feature_params = dict(maxCorners=150,
                                   qualityLevel=0.15,
                                   minDistance=15,
                                   blockSize=7)
        
        self.prev_gray = None
        self.p0 = None
        self.prev_time = 0

    def estimate_velocity(self, current_img, height_m):
        if height_m < 0.3: return 0.0, 0.0

        curr_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        curr_time = time.time()
        
        # Takip noktaları azaldıysa yenile
        if self.prev_gray is None or self.p0 is None or len(self.p0) < 15:
            self.prev_gray = curr_gray
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            self.prev_time = curr_time
            return 0.0, 0.0

        dt = curr_time - self.prev_time
        if dt < 0.001: return 0.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.p0, None, **self.lk_params)

        vx, vy = 0.0, 0.0

        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            
            if len(good_new) > 0:
                flows = good_new - good_old 
                
                # Jitter azaltmak için MEDIAN kullanıyoruz (Ortalama yerine)
                flow_x = np.median(flows[:, 0])
                flow_y = np.median(flows[:, 1])
                
                # Piksel hızı (m/s)
                vx = (flow_x * height_m) / (self.focal_length * dt)
                vy = (flow_y * height_m) / (self.focal_length * dt)

            self.prev_gray = curr_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            self.prev_time = curr_time
        else:
            self.prev_gray = None
            
        return vx, vy

class VisualVelocityNode(Node):
    def __init__(self):
        super().__init__('visual_velocity_estimator')

        # --- PARAMETRELER (TUNE EDECEĞİN YERLER) ---
        
        # 1. Focal Length (Teorik değer)
        self.declare_parameter('focal_length', 370.0)
        
        # 2. Scale Factor (KALİBRASYON)
        # Eğer gerçek hız -5, ölçülen -3 ise -> Scale Factor = 5/3 = 1.66 yapmalısın.
        self.declare_parameter('scale_factor', 1.66) 
        
        # 3. Yayın Frekansı (Hz) - Veri selini önlemek için
        self.declare_parameter('publish_rate', 10.0)
        
        # 4. Low Pass Filter Alpha (0.0 - 1.0)
        # 1.0: Filtre yok (Çok gürültülü)
        # 0.1: Çok yumuşak (Gecikmeli)
        # 0.3 - 0.5 arası genelde iyidir.
        self.declare_parameter('lpf_alpha', 0.2)

        # Değerleri al
        f_len = self.get_parameter('focal_length').value
        self.scale_factor = self.get_parameter('scale_factor').value
        pub_rate = self.get_parameter('publish_rate').value
        self.alpha = self.get_parameter('lpf_alpha').value
        self.alpha_lateral = 0.1
        
        # Hesaplama aralığı (saniye)
        self.process_interval = 1.0 / pub_rate
        self.last_process_time = 0

        self.estimator = OpticalFlowEstimator(focal_length_px=f_len)
        self.bridge = CvBridge()

        # Durum Değişkenleri
        self.current_altitude = 0.0
        self.current_heading = 0.0
        self.is_heading_valid = False
        
        # Filtre için önceki değerler
        self.prev_vn = 0.0
        self.prev_ve = 0.0

        self.prev_body_vx = 0.0
        self.prev_body_vy = 0.0

        qos_fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, '/camera/image', self.image_callback, qos_fast)
        self.create_subscription(Float64, '/mavros/global_position/compass_hdg', self.heading_callback, qos_fast)

        self.vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_enu', 10)
        self.body_vel_pub = self.create_publisher(TwistStamped, '/visual_velocity_body', 10)


        self.get_logger().info(f"Başlatıldı. Scale: {self.scale_factor}, Alpha: {self.alpha}, Rate: {pub_rate}Hz")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)

    def alt_cb(self, msg: Float64):
        self.current_altitude = msg.data
    
    def heading_callback(self, msg: Float64):
        self.current_heading = msg.data
        self.is_heading_valid = True

    def image_callback(self, msg):
        # --- RATE LIMITER ---
        # Kameradan 60 FPS gelse bile biz sadece ayarlanan sürede bir işlem yaparız.
        now = time.time()
        if (now - self.last_process_time) < self.process_interval:
            return
        self.last_process_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            h = self.current_altitude
            if h < 0.5: h = 0.5 

            # 1. Ham Piksel Hızları
            v_px_x, v_px_y = self.estimator.estimate_velocity(cv_image, h)

            # 2. Gövde Hızlarına Çevir (Body Frame)
            # Kamera Montajına göre: Flow Y(+) -> İleri, Flow X(-) -> Sağ
            v_body_fwd = v_px_y
            v_body_right = -v_px_x

            # --- SCALE FACTOR (KALİBRASYON) ---
            # Hatalı ölçümü düzeltmek için çarpan
            v_body_fwd *= self.scale_factor
            v_body_right *= -self.scale_factor

            
            v_body_x_filt = (self.alpha * v_body_fwd) + ((1.0 - self.alpha) * self.prev_body_vx)
            v_body_y_filt = (self.alpha_lateral * v_body_right) + ((1.0 - self.alpha_lateral) * self.prev_body_vy)
            
            self.prev_body_vx = v_body_x_filt
            self.prev_body_vy = v_body_y_filt

            if not self.is_heading_valid:
                return 

            # 3. ENU Dönüşümü
            hdg_rad = math.radians(self.current_heading)
            
            # Raw (Filtresiz) ENU değerleri
            v_north_filt = v_body_x_filt * math.cos(hdg_rad) + self.prev_body_vy * math.sin(hdg_rad)
            v_east_filt = v_body_y_filt * math.sin(hdg_rad) - self.prev_body_vy * math.cos(hdg_rad)


            """ 
            # --- LOW PASS FILTER (TİTREŞİM ENGELLEME) ---
            # Yeni = Alpha * Ham + (1-Alpha) * Eski
            v_north_filt = (self.alpha * v_north_raw) + ((1.0 - self.alpha) * self.prev_vn)
            v_east_filt = (self.alpha * v_east_raw) + ((1.0 - self.alpha) * self.prev_ve)
            """
            
            # body yayınla
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = "map" 
            
            twist.twist.linear.x = float(v_body_x_filt)
            twist.twist.linear.y = float(v_body_y_filt)
            
            self.body_vel_pub.publish(twist)

            # 4. Yayınla
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = "map" 
            
            twist.twist.linear.x = float(v_east_filt)
            twist.twist.linear.y = float(v_north_filt)
            
            self.vel_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f"Hata: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
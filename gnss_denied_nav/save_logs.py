#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import csv
import time
import os
from datetime import datetime
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class BenchmarkLogger(Node):
    def __init__(self):
        super().__init__('benchmark_logger')
        
        # --- AYARLAR ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f'benchmark_data_{timestamp}.csv'
        self.file_path = os.path.join(os.getcwd(), self.filename)
        
        # CSV Dosyası Başlatma
        self.csv_file = open(self.file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Sütunlar: Zaman, Gerçek X, Gerçek Y, Tahmin X, Tahmin Y
        self.csv_writer.writerow(['timestamp', 'true_x', 'true_y', 'est_x', 'est_y'])
        
        self.get_logger().info(f"Kayıt başladı: {self.filename}")
        self.get_logger().info("Topic bekleniyor: /odometry ve /visual_pose_enu")

        # --- DURUM DEĞİŞKENLERİ ---
        self.start_offset_x = 0.0
        self.start_offset_y = 0.0
        self.is_initialized = False # İlk odom verisi alındı mı?
        
        self.true_x = 0.0
        self.true_y = 0.0
        self.est_x = 0.0
        self.est_y = 0.0

        # --- ABONELİKLER ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # 1. Gerçek Konum (Gazebo Doğrudan Odometry)
        # Topic adı: /odometry
        self.create_subscription(Odometry, '/odometry', self.odom_cb, qos_profile)
        
        # 2. Tahmin Edilen Konum (Bizim Algoritma)
        self.create_subscription(PoseStamped, '/visual_pose_enu', self.visual_cb, 10)

        # --- KAYIT ZAMANLAYICISI (10 Hz) ---
        self.timer = self.create_timer(0.1, self.log_data)

    def odom_cb(self, msg):
        # Odometry mesajından pozisyonu al
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y

        # İlk gelen veriyi "Sıfır Noktası" (Offset) olarak kaydet
        if not self.is_initialized:
            self.start_offset_x = raw_x
            self.start_offset_y = raw_y
            self.is_initialized = True
            self.get_logger().info(f"Referans Noktası Alındı (Gazebo): X={raw_x:.2f}, Y={raw_y:.2f}")

        # Başlangıç farkını çıkararak (0,0)'a oturt
        self.true_x = raw_x - self.start_offset_x
        self.true_y = raw_y - self.start_offset_y

    def visual_cb(self, msg):
        # Visual Positioning zaten (0,0)'dan başladığı için direkt alıyoruz
        self.est_x = msg.pose.position.x
        self.est_y = msg.pose.position.y

    def log_data(self):
        # Başlangıç verisi gelmeden kaydetme
        if not self.is_initialized:
            return

        t = time.time()
        self.csv_writer.writerow([t, self.true_x, self.true_y, self.est_x, self.est_y])
        self.csv_file.flush() 

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float64  # Heading verisi genelde Float64 olarak gelir
from cv_bridge import CvBridge
import cv2
import os
import time
import json
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

"""
Description : 
---
Bu düğüm (node), drone kamerasından görüntü alır ve MAVROS üzerinden
gelen GPS ve Heading (Pusula) verileriyle eşleştirerek kaydeder.

İşlevleri:
1. /camera/image üzerinden görüntü akışını alır.
2. /mavros/global_position/global üzerinden GPS verisini alır.
3. /mavros/global_position/compass_hdg üzerinden Heading (Yön) verisini alır.
4. Belirlenen zaman aralığında (3 FPS) görüntüyü diske yazar.
5. Her görüntü için bir metadata (JSON) kaydı oluşturur.
---
"""

class ImageSaver(Node):
    """
    Görüntü ve telemetri verilerini senkronize bir şekilde kaydeden ROS 2 Düğümü.
    """
    def __init__(self):
        super().__init__("image_saver")

        # --- 1. Dosya Sistemi Ayarları ---
        # Her çalıştırmada üzerine yazmamak için zaman damgalı klasör oluşturuyoruz
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.expanduser(f"~/frames_{self.timestamp}")
        
        # Klasör yoksa oluştur
        os.makedirs(self.output_dir, exist_ok=True)

        # Metadata verilerini tutacak JSON dosyasının yolu
        self.json_path = os.path.join(self.output_dir, "frames_details.json")
        self.frames_data = [] # Bellekte tutulan veri listesi

        # --- 2. Gerekli Araçlar ---
        # ROS görüntü mesajlarını OpenCV formatına çevirmek için köprü
        self.bridge = CvBridge()
        self.frame_count = 0 # Kaydedilen kare sayacı

        # --- 3. Telemetri Değişkenleri ---
        # GPS verisi (Enlem, Boylam, İrtifa) - Başlangıçta None
        self.current_gps = {"lat": None, "lon": None, "alt": None}
        # Heading (Pusula Yönü 0-360 derece) - Başlangıçta None
        self.current_heading = None

        # --- 4. QoS Ayarları (Quality of Service) ---
        # MAVROS genelde 'Best Effort' yayın yapar, bu yüzden Reliable yerine
        # Best Effort kullanmak veri kaybını veya eşleşmeme sorununu önler.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- 5. Abonelikler (Subscribers) ---
        
        # Kamera Görüntüsü Aboneliği
        self.create_subscription(
            Image, 
            "/camera/image", 
            self.image_callback, 
            10
        )

        # GPS Konum Aboneliği (MAVROS)
        self.create_subscription(
            NavSatFix, 
            "/mavros/global_position/global", 
            self.gps_callback, 
            qos_profile=qos
        )

        # YENİ: Heading (Pusula) Aboneliği (MAVROS)
        # Genellikle Float64 tipinde, derece cinsinden veri döner.
        self.create_subscription(
            Float64,
            "/mavros/global_position/compass_hdg",
            self.heading_callback,
            qos_profile=qos
        )

        # --- 6. Zamanlayıcı (Timer) ---
        # Görüntü kaydetme frekansı: Saniyede 3 kare (1/3 saniye aralıkla)
        self.timer = self.create_timer(1.5, self.save_frame)

        # Son alınan görüntüyü saklamak için değişken
        self.latest_image = None

        self.get_logger().info(f"Görüntü kaydedici başlatıldı. Kayıt Yeri: {self.output_dir}")

    def gps_callback(self, msg: NavSatFix):
        """
        GPS verisi geldiğinde çağrılır. Enlem, boylam ve irtifayı günceller.
        """
        self.current_gps["lat"] = msg.latitude
        self.current_gps["lon"] = msg.longitude
        self.current_gps["alt"] = msg.altitude

    def heading_callback(self, msg: Float64):
        """
        Heading verisi geldiğinde çağrılır. Drone'un burnunun baktığı yönü günceller.
        msg.data: 0 ile 360 derece arasında bir float değerdir.
        """
        self.current_heading = msg.data

    def image_callback(self, msg: Image):
        """
        Kamera verisi geldiğinde çağrılır. ROS mesajını OpenCV formatına çevirir.
        """
        try:
            # ROS Image -> OpenCV Image (BGR formatında)
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüştürme hatası: {str(e)}")

    def save_frame(self):
        """
        Zamanlayıcı tarafından tetiklenir. Mevcut görüntüyü ve o anki telemetriyi kaydeder.
        """
        # Eğer henüz hiç görüntü gelmediyse işlem yapma
        if self.latest_image is None:
            return

        # Dosya isimlendirme (timestamp + frame numarası)
        filename = f"frame_{self.timestamp}_{self.frame_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Görüntüyü diske yaz (OpenCV)
        cv2.imwrite(filepath, self.latest_image)

        # Metadata sözlüğünü oluştur
        frame_info = {
            "filename": filename,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gps": self.current_gps.copy(),  # Sözlüğü kopyalayarak al (referans hatası olmaması için)
            "heading": self.current_heading  # YENİ: Heading bilgisini ekle
        }
        
        # Listeye ekle
        self.frames_data.append(frame_info)

        # JSON dosyasını güncelle (Her karede yazmak güvenlidir ama performans için aralık verilebilir)
        # Veri kaybını önlemek için 'w' moduyla her seferinde güncel listeyi yazıyoruz.
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.frames_data, f, indent=4)
        except Exception as e:
            self.get_logger().error(f"JSON yazma hatası: {e}")

        self.frame_count += 1
        
        # Loglama: Her 10 karede bir terminale bilgi ver
        if self.frame_count % 10 == 0:
            hdg_str = f"{self.current_heading:.2f}" if self.current_heading else "Yok"
            self.get_logger().info(f"Kaydedildi: {self.frame_count} | Dosya: {filename} | Hdg: {hdg_str}")

    def destroy_node(self):
        """
        Node kapatılırken son işlemleri yapar.
        """
        # Çıkarken JSON kaydını garanti et
        if self.frames_data:
            with open(self.json_path, "w") as f:
                json.dump(self.frames_data, f, indent=4)
        
        self.get_logger().info(f"Kapatılıyor. Toplam {self.frame_count} frame kaydedildi.")
        self.get_logger().info(f"JSON dosyası: {self.json_path}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        # Node'u çalışır durumda tut
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Klavye ile durduruldu (Ctrl+C). Çıkış yapılıyor...")
    finally:
        # Temiz bir kapanış yap
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
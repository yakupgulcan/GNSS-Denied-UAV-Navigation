#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.msg import OverrideRCIn, State
from std_msgs.msg import Float64, String
from mavros_msgs.srv import CommandBool, SetMode
import time

class AltHoldPController(Node):
    def __init__(self):
        super().__init__('althold_p_controller')

        # --- AYARLAR ---
        self.TARGET_ALTITUDE = 29.0  # Hedef: 15 metre
        self.KP_ALTITUDE = 12.0      # P Katsayısı (Hata * Kp = PWM Artışı)
        self.HOVER_PWM = 1500        # Nötr Gaz (Sabit İrtifa)
        self.MAX_PWM = 1750          # Maksimum Tırmanma PWM'i
        self.MIN_PWM = 1250          # Maksimum Alçalma PWM'i
        self.ACCEPTANCE_ERROR = 1.0  # 20cm hata payı kabul edilir

        # Değişkenler
        self.takeoff_achieved = False
        self.current_altitude = 0.0
        self.mavros_state = State()
        self.is_target_reached = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # --- ABONELİKLER ---
        # Yükseklik Verisi (Barometre veya Lidar - Relative Altitude)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.alt_cb, qos)
        # Durum Verisi
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)

        # --- YAYINCILAR ---
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.status_pub = self.create_publisher(String, '/althold_ctrl/status', 10)

        # --- SERVİSLER ---
        self.client_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.client_mode = self.create_client(SetMode, '/mavros/set_mode')

        # --- DÖNGÜ (20 Hz) ---
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("AltHold P-Kontrolcü Başlatıldı. Hedef: 15m")

    # --- CALLBACKS ---
    def alt_cb(self, msg: Float64):
        self.current_altitude = msg.data

    def state_cb(self, msg: State):
        self.mavros_state = msg

    # --- P KONTROLCÜ FONKSİYONU ---
    def calculate_throttle_command(self):
        """
        Basit P Kontrolcü:
        Hata = Hedef - Mevcut
        PWM = 1500 + (Hata * Kp)
        """
        error = self.TARGET_ALTITUDE - self.current_altitude

        # Hata tolerans içindeyse (Örn: 14.8m - 15.2m arası)
        if abs(error) < self.ACCEPTANCE_ERROR:
            self.is_target_reached = True
            return self.HOVER_PWM # 1500 gönder, irtifayı kilitle.

        # P Hesabı
        control_output = error * self.KP_ALTITUDE
        
        # Base PWM üzerine ekle
        target_pwm = self.HOVER_PWM + control_output

        # Sınırlandırma (Saturation)
        # PWM çok yüksek veya çok düşük olmamalı
        target_pwm = max(self.MIN_PWM, min(self.MAX_PWM, target_pwm))

        return int(target_pwm)

    # --- ANA DÖNGÜ ---
    def control_loop(self):
        if self.takeoff_achieved:
            self.set_mode("ALT_HOLD")
            return
        # Bağlantı yoksa bekle
        if not self.mavros_state.connected:
            return

        # 1. Önce ARM ve MOD kontrolü (Otomatik Hazırlık)
        if self.mavros_state.mode != "STABILIZE":
            self.set_mode("STABILIZE")
            self.send_rc_override(1500) # Mod değişirken nötr kal
            return
        
        if not self.mavros_state.armed:
            self.arm_vehicle()
            self.send_rc_override(1000) # Arm olurken gazı kes
            return

        # 2. P Kontrolcü ile Yükselme
        throttle_pwm = self.calculate_throttle_command()
        
        # RC Mesajını gönder
        self.send_rc_override(throttle_pwm)

        # Durum Yayını

        status_msg = f"Alt: {self.current_altitude:.2f}m | Err: {(self.TARGET_ALTITUDE - self.current_altitude):.2f}m | PWM: {throttle_pwm}"
        self.get_logger().info(status_msg)
        
        if self.is_target_reached:
            # Sadece bilgi logu, döngü devam eder (hover tutmak için)
            self.takeoff_achieved = True
            pass 

    # --- YARDIMCI METODLAR ---
    def send_rc_override(self, throttle_pwm):
        msg = OverrideRCIn()
        msg.channels = [65535] * 18
        msg.channels[0] = 1500 # Roll
        msg.channels[1] = 1500 # Pitch
        msg.channels[2] = int(throttle_pwm) # Throttle (P Kontrolcü Çıkışı)
        msg.channels[3] = 1500 # Yaw
        self.rc_pub.publish(msg)

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.client_mode.call_async(req)

    def arm_vehicle(self):
        req = CommandBool.Request()
        req.value = True
        self.client_arm.call_async(req)

def main():
    rclpy.init()
    node = AltHoldPController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_rc_override(1500)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
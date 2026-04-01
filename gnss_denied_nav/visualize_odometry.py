import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.interpolate import interp1d

class VisualReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.df_real = None
        self.df_vis = None
        
    def load_data(self):
        if not os.path.exists(self.filename):
            print(f"Hata: Dosya bulunamadı -> {self.filename}")
            return False
            
        try:
            self.df = pd.read_csv(self.filename)
            print(f"Veri yüklendi. Toplam satır: {len(self.df)}")
            
            # Verileri ayır
            self.df_real = self.df[self.df['type'] == 'REAL'].sort_values('timestamp')
            self.df_vis = self.df[self.df['type'] == 'VISUAL'].sort_values('timestamp')
            
            if self.df_real.empty or self.df_vis.empty:
                print("Hata: REAL veya VISUAL veri kümelerinden biri boş.")
                return False
                
            return True
        except Exception as e:
            print(f"CSV okuma hatası: {e}")
            return False

    def remove_outliers_iqr(self, df, column, factor=1.5):
        """Basit IQR outlier temizliği"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    def synchronize_and_calculate_error(self):
        """
        Görsel verilerin zaman damgalarına karşılık gelen Gerçek Konumları
        interpolasyon (ara değer bulma) ile hesaplar.
        """
        # Zamanı başlangıca göre sıfırla (saniye cinsinden)
        start_time = min(self.df_real['timestamp'].min(), self.df_vis['timestamp'].min())
        
        t_real = self.df_real['timestamp'] - start_time
        x_real = self.df_real['x']
        y_real = self.df_real['y']
        
        t_vis = self.df_vis['timestamp'] - start_time
        x_vis = self.df_vis['x']
        y_vis = self.df_vis['y']

        # İnterpolasyon fonksiyonları oluştur (Zamana bağlı X ve Y)
        # fill_value="extrapolate" ile veri sınırları dışındaki ufak kaymaları tahmin et
        f_x = interp1d(t_real, x_real, kind='linear', fill_value="extrapolate")
        f_y = interp1d(t_real, y_real, kind='linear', fill_value="extrapolate")

        # Görsel verinin zamanına karşılık gelen Gerçek (Ground Truth) konumu bul
        x_truth_at_vis_time = f_x(t_vis)
        y_truth_at_vis_time = f_y(t_vis)

        # Hata hesapla (Öklid Mesafesi)
        errors = np.sqrt((x_vis - x_truth_at_vis_time)**2 + (y_vis - y_truth_at_vis_time)**2)
        
        return t_vis, x_vis, y_vis, x_real, y_real, errors

    def generate_report(self):
        if not self.load_data():
            return

        # Veriyi işle
        t_vis, x_vis, y_vis, x_real, y_real, errors = self.synchronize_and_calculate_error()
        
        # --- KRİTİK DÜZELTME: Verileri NumPy dizisine çevir ---
        # Pandas Series formatı matplotlib'de hataya yol açtığı için 
        # .to_numpy() ile saf sayısal diziye çeviriyoruz.
        x_real_np = x_real.to_numpy()
        y_real_np = y_real.to_numpy()
        x_vis_np = x_vis.to_numpy()
        y_vis_np = y_vis.to_numpy()
        # ------------------------------------------------------

        # İstatistikler
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print("-" * 30)
        print("     PERFORMANS RAPORU")
        print("-" * 30)
        print(f"İşlenen Kare Sayısı : {len(x_vis)}")
        print(f"Ortalama Hata       : {mean_error:.2f} metre")
        print(f"RMSE (Karesel Ort.) : {rmse:.2f} metre")
        print(f"Maksimum Hata       : {max_error:.2f} metre")
        print("-" * 30)

        # --- GÖRSELLEŞTİRME ---
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)

        # 1. Yörünge Grafiği (Trajectory) - Sol Taraf (Büyük)
        ax1 = fig.add_subplot(gs[:, 0])
        
        # Düzeltilmiş NumPy dizilerini kullanıyoruz:
        ax1.plot(x_real_np, y_real_np, 'k-', linewidth=2, label='Gerçek Rota (GPS)', alpha=0.7)
        ax1.plot(x_vis_np, y_vis_np, 'r--.', markersize=4, linewidth=1, label='ORB Görsel Konum')
        
        # Başlangıç ve Bitiş Noktaları (NumPy indeksleme ile [0] ve [-1])
        ax1.scatter(x_real_np[0], y_real_np[0], c='g', marker='^', s=100, label='Başlangıç', zorder=5)
        ax1.scatter(x_real_np[-1], y_real_np[-1], c='b', marker='s', s=100, label='Bitiş', zorder=5)
        
        ax1.set_title(f"Yörünge Karşılaştırması (RMSE: {rmse:.2f}m)")
        ax1.set_xlabel("Doğu (East) [m]")
        ax1.set_ylabel("Kuzey (North) [m]")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axis('equal')

        # 2. Zaman İçindeki Hata (Error over Time) - Sağ Üst
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_vis.to_numpy(), errors.to_numpy(), 'r-', alpha=0.7) # Buraya da eklendi
        ax2.fill_between(t_vis.to_numpy(), 0, errors.to_numpy(), color='red', alpha=0.1)
        ax2.set_title("Zaman İçindeki Konum Hatası")
        ax2.set_ylabel("Hata [m]")
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Ortalama hata çizgisi
        ax2.axhline(mean_error, color='blue', linestyle='--', label=f'Ort: {mean_error:.2f}m')
        ax2.legend()

        # 3. Hata Histogramı - Sağ Alt
        ax3 = fig.add_subplot(gs[1, 1])
        # Errors zaten numpy dizisi olduğu için burada sorun yoktu ama garanti olsun
        ax3.hist(errors.to_numpy(), bins=20, color='gray', edgecolor='black', alpha=0.7)
        ax3.set_title("Hata Dağılımı (Histogram)")
        ax3.set_xlabel("Hata Miktarı [m]")
        ax3.set_ylabel("Örnek Sayısı")

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Örnek kullanım: python plot_results.py visual_log_2025-xx-xx.csv
    file_path = '' # Varsayılan dosya adı
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    reporter = VisualReportGenerator(file_path)
    reporter.generate_report()
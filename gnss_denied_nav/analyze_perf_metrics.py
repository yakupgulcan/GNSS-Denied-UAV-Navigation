import pandas as pd
import numpy as np
import sys

def analyze(csv_file):
    df = pd.read_csv(csv_file)
    
    # 1. Ortalama İşlem Süresi ve FPS
    avg_time_ms = df['proc_time_ms'].mean()
    avg_fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    
    # 2. Eşleşme Başarısı
    total_frames = len(df)
    matches = df[df['status'] == 'MATCH']
    match_count = len(matches)
    success_rate = (match_count / total_frames) * 100 if total_frames > 0 else 0
    
    # 3. Konum Hatası (Sadece MATCH olanlar için)
    rmse = np.sqrt((matches['error_m']**2).mean())
    max_error = matches['error_m'].max()
    
    print("-" * 40)
    print(f"ANALİZ RAPORU: {csv_file}")
    print("-" * 40)
    print(f"Toplam Kare Sayısı: {total_frames}")
    print(f"Başarılı Eşleşme  : {match_count} (%{success_rate:.2f})")
    print("-" * 40)
    print(f"Ort. İşlem Süresi : {avg_time_ms:.2f} ms")
    print(f"Ortalama Hız      : {avg_fps:.2f} FPS")
    print("-" * 40)
    print(f"Konum Hatası (RMSE): {rmse:.2f} metre")
    print(f"Maksimum Hata      : {max_error:.2f} metre")
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python3 analyze_results.py <dosya.csv>")
    else:
        analyze(sys.argv[1])
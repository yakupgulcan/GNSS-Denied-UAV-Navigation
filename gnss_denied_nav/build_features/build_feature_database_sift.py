import cv2
import numpy as np
import os
import json

# --- AYARLAR ---
frames_path = "~/frames_2026-01-17_17-22-56"
N_FEATURES = 2000 # SIFT için  özellik iyi bir başlangıçtır

def build_sift_database(frames_dir):
    """
    SIFT özellik veritabanını oluşturur.
    Kaydedilenler: descriptors, keypoints, filenames, gps_data, headings
    """

    # 1. SIFT Başlatıcı
    # SIFT patent süresi dolduğu için modern OpenCV sürümlerinde (4.4+) doğrudan kullanılabilir.
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = []

    json_path = os.path.join(frames_dir, "frames_details.json")
    
    if not os.path.exists(json_path):
        print(f"HATA: JSON dosyası bulunamadı: {json_path}")
        return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"SIFT ile veritabanı oluşturuluyor. Toplam {len(frames_info)} kare...")

    count = 0
    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        
        if not os.path.exists(img_path):
            continue

        # Görüntüyü Oku (Gri Tonlama)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 2. SIFT Özellik Çıkarımı
        kp, des = sift.detectAndCompute(img, None)
        
        # Özellik bulunamazsa atla
        if des is None or len(kp) == 0:
            continue

        # KeyPoint Serileştirme (x, y, size, angle)
        # SIFT keypoint yapısı ORB ile uyumludur, aynı mantığı kullanabiliriz.
        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        
        gps_data.append(entry.get("gps", {}))
        headings_db.append(entry.get("heading", 0.0))
        
        count += 1
        if count % 50 == 0:
            print(f"{count} kare işlendi...")

    # .npz olarak kaydet
    output_path = os.path.join(frames_dir, f"features_db_sift_{N_FEATURES}.npz")
    
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32)
    )

    print("-" * 30)
    print(f"İşlem tamamlandı.")
    print(f"İşlenen Resim Sayısı: {len(filenames)}")
    print(f"Kayıt Yeri: {output_path}")

if __name__ == "__main__":
    real_frames_dir = os.path.expanduser(frames_path)
    build_sift_database(real_frames_dir)
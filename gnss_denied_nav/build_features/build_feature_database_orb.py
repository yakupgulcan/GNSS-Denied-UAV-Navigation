import cv2
import numpy as np
import os
import json
"""
Bu dosyada toplanan görüntüler ve görüntülerle ilgili json dosyası alınır
ORB ile fwatureları çıkarılmış ve npz dosyasına kaydedilmiş şekilde tekrardan
saklanır.

"""
frames_path = "~/frames_2026-01-09_08-47-17"

def build_feature_database(frames_dir):
    """
    ORB özellik veritabanını oluşturur.
    Kaydedilenler: descriptors, keypoints, filenames, gps_data, headings
    """

    # 1. nfeatures 500 olarak ayarlandı (Daha hızlı, ancak daha az detay)
    orb = cv2.ORB_create(nfeatures=2000)

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = [] # YENİ: Heading verisi için liste

    json_path = os.path.join(frames_dir, "frames_details.json")
    
    if not os.path.exists(json_path):
        print(f"JSON dosyası bulunamadı: {json_path}")
        return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"Toplam {len(frames_info)} kare işleniyor...")

    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        
        # Dosya var mı kontrolü
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Özellik çıkarımı
        kp, des = orb.detectAndCompute(img, None)
        
        # Eğer özellik bulunamazsa bu kareyi atla
        if des is None or len(kp) == 0:
            continue

        # KeyPoint'leri serileştirilebilir formata çevir (x, y, size, angle)
        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        
        # GPS verisini al
        gps_data.append(entry.get("gps", {}))
        
        # YENİ: Heading verisini al (Eğer yoksa 0.0 varsay)
        headings_db.append(entry.get("heading", 0.0))

    # .npz olarak kaydet
    output_path = os.path.join(frames_dir, "features_db_2000.npz")
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32) # YENİ: Heading dizisi
    )

    print(f"İşlem tamamlandı. {len(filenames)} görüntü için veritabanı oluşturuldu.")
    print(f"Dosya: {output_path}")

if __name__ == "__main__":
    frames_dir = os.path.expanduser(frames_path)
    build_feature_database(frames_dir)
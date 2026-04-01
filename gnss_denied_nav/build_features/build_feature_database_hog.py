import cv2
import numpy as np
import os
import json
from skimage.feature import hog

# --- EN YÜKSEK AYARLAR (MAX DETAIL) ---
# Bu ayarlar hog_detect_match.py ile BİREBİR AYNI OLMALI
TARGET_WIDTH = 480
TARGET_HEIGHT = 360

HOG_ORIENTATIONS = 9
# (8, 8) hücre boyutu, (16, 16)'ya göre çok daha ince detayları yakalar.
HOG_PIXELS_PER_CELL = (16, 16) # Yüksek Detay
HOG_CELLS_PER_BLOCK = (2, 2)

frames_path = "~/frames_2026-01-09_08-47-17"

def build_hog_database(frames_dir):
    hog_features_db = []
    filenames = []
    gps_data = []
    headings_db = []

    json_path = os.path.join(frames_dir, "frames_details.json")
    
    if not os.path.exists(json_path):
        print(f"HATA: {json_path} yok.")
        return

    with open(json_path) as f:
        frames_info = json.load(f)

    print("-" * 40)
    print(f"YÜKSEK DETAYLI HOG Veritabanı Oluşturuluyor...")
    print(f"Hedef Boyut: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Hücre Boyutu: {HOG_PIXELS_PER_CELL} (Daha küçük = Daha hassas)")
    print("-" * 40)

    count = 0
    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # 1. RESIZE (Garanti olsun diye 640x480'e zorluyoruz)
        img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

        # 2. HOG HESAPLA (Yüksek Detay Ayarlarıyla)
        try:
            features = hog(img_resized, 
                           orientations=HOG_ORIENTATIONS, 
                           pixels_per_cell=HOG_PIXELS_PER_CELL,
                           cells_per_block=HOG_CELLS_PER_BLOCK, 
                           visualize=False, 
                           feature_vector=True)
        except Exception as e:
            print(f"Hata: {e}")
            continue

        if features is None: continue

        hog_features_db.append(features)
        filenames.append(entry["filename"])
        gps_data.append(entry.get("gps", {}))
        headings_db.append(entry.get("heading", 0.0))
        
        count += 1
        if count % 50 == 0: print(f"{count} işlendi...") # Daha sık bilgi ver

    if not hog_features_db:
         print("Hata: Özellik çıkarılamadı.")
         return

    # Kaydet (İsim maxres oldu)
    output_path = os.path.join(frames_dir, "features_db_hog_480_360_16.npz")
    
    np.savez_compressed(
        output_path,
        hog_features=np.array(hog_features_db, dtype=np.float32),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32)
    )
    print("=" * 40)
    print(f"TAMAMLANDI! Toplam {len(filenames)} görüntü işlendi.")
    # Vektör boyutu (8,8) ayarıyla bayağı büyüyecektir (tahmini 15000+), bu normaldir.
    print(f"HOG Vektör Boyutu (Hassasiyet): {len(hog_features_db[0])}") 
    print(f"Dosya: {output_path}")
    print("=" * 40)

if __name__ == "__main__":
    real_frames_dir = os.path.expanduser(frames_path)
    build_hog_database(real_frames_dir)
#!/usr/bin/env python3
"""
Bu script bir görsel üzerinde SIFT ve ORB Keypoint'lerini karşılaştırmalı olarak gösterir.
Kullanıcı herhangi bir resim yolunu vererek testi yapabilir.
"""

import cv2
import numpy as np
import argparse

def draw_keypoints_side_by_side(image_path):
    # Görseli Oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"HATA: Görüntü okunamadı -> {image_path}")
        return

    # İşlenebilir boyuta getir (Çok büyükse küçült)
    MAX_WIDTH = 1200
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * scale)))

    # --- 1. SIFT ---
    # nfeatures=1000 yaparak ORB ile adil bir yarış başlatalım
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_sift, _ = sift.detectAndCompute(img, None)
    
    # SIFT Görseli (Kırmızı Noktalar)
    # cv2.drawKeypoints fonksiyonu noktaları çizer.
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS -> Boyut ve yönü de gösterir (Yuvarlaklar)
    img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Bilgi Yaz
    cv2.putText(img_sift, f"SIFT: {len(kp_sift)} features", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 2. ORB ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp_orb, _ = orb.detectAndCompute(img, None)
    
    # ORB Görseli (Yeşil Noktalar)
    img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0), flags=0) # flags=0 sadece nokta
    
    # Bilgi Yaz
    cv2.putText(img_orb, f"ORB: {len(kp_orb)} features", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 3. BİRLEŞTİR VE GÖSTER ---
    # Yan yana koy
    combined = np.hstack((img_sift, img_orb))
    
    # Dosyaya kaydet (Göremeyebilirsin ama dosya oluşur)
    output_file = "comparison_sift_vs_orb.jpg"
    cv2.imwrite(output_file, combined)
    print(f"Karşılaştırma kaydedildi: {output_file}")
    print(f"SIFT Nokta Sayısı: {len(kp_sift)}")
    print(f"ORB Nokta Sayısı : {len(kp_orb)}")

if __name__ == "__main__":
    # Test için varsayılan bir yol veya argüman
    # Kullanım: python3 compare_features.py --image /path/to/image.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    draw_keypoints_side_by_side(args.image)

import cv2
import numpy as np
import os

TARGET_WIDTH = 640

class AKAZEDetectAndMatch:
    def __init__(self, db_path, threshold=0.001): # Threshold varsayılan olarak artırıldı
        # Veritabanı oluştururken kullanılan parametrelerin aynısı
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            threshold=threshold,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        self.db_path = db_path
        print(f"[AKAZE] Yükleniyor: {db_path} ...")
        
        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_descriptors = self.db['descriptors']
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']
        except Exception as e:
            print(f"[AKAZE] Hata: {e}")
            return

        # Flattening
        self.all_descriptors = []
        self.desc_to_img_id = []
        temp_desc = []
        temp_ids = []
        
        for img_id, desc in enumerate(self.db_descriptors):
            if desc is not None and len(desc) > 0:
                temp_desc.append(desc)
                temp_ids.append(np.full(len(desc), img_id, dtype=np.int32))

        if temp_desc:
            self.all_descriptors = np.vstack(temp_desc)
            self.desc_to_img_id = np.concatenate(temp_ids)
        
        # FLANN (LSH)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50) # Biraz daha hassasiyet için artırdık
        
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.matcher.add([self.all_descriptors])
        self.matcher.train()
        print(f"[AKAZE] Hazır. Toplam özellik: {len(self.all_descriptors)}")

    def _preprocess(self, frame):
        # Resize + CLAHE
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
        
        # CLAHE uygula (Kontrast dengeleme)
        return self.clahe.apply(frame)

    def get_location(self, frame, min_match_count=8, top_k=5):
        frame_proc = self._preprocess(frame)
        kp, des = self.akaze.detectAndCompute(frame_proc, None)
        
        if des is None or len(des) < 5: return []

        matches = self.matcher.knnMatch(des, k=2)
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            # AKAZE için ratio test 0.75 iyidir
            if m.distance < 0.75 * n.distance:
                good_matches_indices.append(m.trainIdx)

        if len(good_matches_indices) < min_match_count: return []

        matched_img_ids = self.desc_to_img_id[np.array(good_matches_indices)]
        counts = np.bincount(matched_img_ids)
        best_indices = np.argsort(counts)[::-1]
        
        candidates = []
        for idx in best_indices[:top_k]:
            score = counts[idx]
            if score < min_match_count: break
            candidates.append((self.gps_data[idx], self.filenames[idx], int(score)))

        return candidates

    def match_frame_to_descriptors(self, frame, target_descriptors, min_match_count=8):
        """
        Gelişmiş Backtracking (RANSAC + CLAHE)
        """
        # 1. Ön İşleme (Resize + CLAHE)
        frame_proc = self._preprocess(frame)
        h, w = frame_proc.shape[:2]
        center_x, center_y = w / 2, h / 2

        # 2. Özellik Çıkarımı
        kp_curr, des_curr = self.akaze.detectAndCompute(frame_proc, None)
        if des_curr is None or len(des_curr) < 5 or target_descriptors is None: return None

        # 3. Eşleştirme (Brute Force - Hamming)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)
        
        # Başlangıç filtresi: En iyi mesafeler
        matches = sorted(matches, key=lambda x: x.distance)
        # Sadece iyi eşleşmeleri al (Tüm eşleşmelerin ilk %30'u veya max 50)
        # Bu gürültüyü azaltır.
        top_n = min(50, int(len(matches) * 0.3))
        good_matches = matches[:top_n]

        if len(good_matches) < min_match_count: return None

        # 4. KRİTİK: RANSAC ile Geometrik Doğrulama
        # Hedef resmin keypointlerine sahip olmadığımız için (sadece descriptor var),
        # Tam homography yapamayız ancak "Perspective Transform" yerine 
        # Outlier eliminasyonu için basit istatistik kullanabiliriz.
        # Veya build_database'de keypointleri de kaydedip burada kullanabilirdik.
        # Şimdilik mevcut yapı üzerinden "Robust Statistics" ile gidelim.

        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        
        # Outlier Temizliği (Basit İstatistiksel Yöntem)
        # Ortalama etrafındaki noktaları al, çok uzak olanları at.
        mean = np.mean(src_pts, axis=0)
        std = np.std(src_pts, axis=0)
        
        # 2 Standart sapma dışındakileri at (Z-Score Filtering)
        filtered_pts = []
        for pt in src_pts:
            if abs(pt[0] - mean[0]) < 2 * std[0] and abs(pt[1] - mean[1]) < 2 * std[1]:
                filtered_pts.append(pt)
        
        if len(filtered_pts) < min_match_count: return None
        
        filtered_pts = np.array(filtered_pts)

        # 5. Sonuç Hesaplama (Temizlenmiş noktalarla)
        centroid_x = np.mean(filtered_pts[:, 0])
        centroid_y = np.mean(filtered_pts[:, 1])
        
        error_x = centroid_x - center_x
        error_y = centroid_y - center_y
        spread_x = np.std(filtered_pts[:, 0])

        # Skor olarak temizlenmiş nokta sayısını döndür
        return len(filtered_pts), error_x, error_y, spread_x, centroid_x, centroid_y
import cv2
import numpy as np
import os

# Parametreler
TARGET_WIDTH = 640 # Görüntü işleme genişliği (Hız için)

class SIFTDetectAndMatch:
    def __init__(self, db_path, n_features=500):
        # 1. SIFT Başlatıcı
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.db_path = db_path
        
        print(f"[SIFT] Veritabanı yükleniyor: {db_path} ...")
        
        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_descriptors = self.db['descriptors']
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']
        except Exception as e:
            print(f"[SIFT] Veritabanı hatası: {e}")
            # Hata durumunda boş listelerle devam et
            self.db_descriptors = []
            self.filenames = []
            self.gps_data = []

        # --- Veritabanı Düzleştirme (Flattening) ---
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
            
            # --- FLANN Parametreleri (SIFT İÇİN KRİTİK) ---
            # SIFT (Float) için KD-Tree kullanılır.
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # Checks: Hassasiyet/Hız dengesi.
            search_params = dict(checks=50) 
            
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.matcher.add([self.all_descriptors])
            self.matcher.train()
            print(f"[SIFT] Model hazır! Toplam {len(self.all_descriptors)} özellik indekslendi.")
        else:
            print("[SIFT] UYARI: Veritabanında geçerli özellik bulunamadı!")
            self.matcher = None

    def _resize_frame(self, frame):
        """Yardımcı fonksiyon: Görüntüyü oran koruyarak küçültür."""
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
        return frame

    def detectAndCompute(self, frame, mask=None):
        """
        ROS Visual Estimator Node uyumluluğu için standart arayüz.
        Breadcrumb (Ekmek kırıntısı) kaydederken kullanılır.
        """
        # Tutarlılık için burada da resize yapıyoruz.
        frame_resized = self._resize_frame(frame)
        return self.sift.detectAndCompute(frame_resized, mask)

    def get_location(self, frame, min_match_count=8, top_k=5):
        """
        Genel konumlandırma için veritabanı taraması.
        """
        if self.matcher is None: return []

        frame = self._resize_frame(frame)

        # Özellik Çıkarımı
        kp, des = self.sift.detectAndCompute(frame, None)
        
        if des is None or len(des) < 5:
            return []

        # Eşleştirme (KNN)
        matches = self.matcher.knnMatch(des, k=2)

        # Ratio Test (Lowe's Ratio Test)
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            # SIFT için 0.7 veya 0.75 idealdir
            if m.distance < 0.75 * n.distance:
                good_matches_indices.append(m.trainIdx)

        if len(good_matches_indices) < min_match_count:
            return []

        # Oylama
        matched_img_ids = self.desc_to_img_id[np.array(good_matches_indices)]
        counts = np.bincount(matched_img_ids)
        
        # En iyi adayları sırala
        best_indices = np.argsort(counts)[::-1]
        
        candidates = []
        for idx in best_indices[:top_k]:
            score = counts[idx]
            if score < min_match_count:
                break
            candidates.append((self.gps_data[idx], self.filenames[idx], int(score)))

        return candidates

    def match_frame_to_descriptors(self, frame, target_descriptors, min_match_count=8):
        """
        BACKTRACKING İÇİN EŞLEŞTİRME:
        Anlık kamerayı, breadcrumb_stack'teki kayıtlı descriptorlar ile karşılaştırır.
        """
        # 1. Resize
        frame = self._resize_frame(frame)
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2

        # 2. Anlık Özellik Çıkarımı
        kp_curr, des_curr = self.sift.detectAndCompute(frame, None)
        
        if des_curr is None or len(des_curr) < 5 or target_descriptors is None:
            return None

        # 3. Eşleştirme (Brute Force - NORM_L2)
        # SIFT float descriptor ürettiği için NORM_L2 kullanılır.
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)
        
        # Mesafeye göre sırala
        matches = sorted(matches, key=lambda x: x.distance)
        
        # En iyi 50 eşleşmeyi al
        good_matches = matches[:50]

        if len(good_matches) < min_match_count:
            return None

        # 4. Hata Hesaplama
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        
        centroid_x = np.mean(src_pts[:, 0])
        centroid_y = np.mean(src_pts[:, 1])
        
        error_x = centroid_x - center_x
        error_y = centroid_y - center_y
        
        spread_x = np.std(src_pts[:, 0])

        return len(good_matches), error_x, error_y, spread_x, centroid_x, centroid_y
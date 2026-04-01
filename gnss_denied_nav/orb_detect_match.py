import cv2
import numpy as np
import os


# GLOBAL PARAMETRELER - Declare_parameter yapacağız sonradan VE YAML
N_FEATURES = 2000
class ORBDetectAndMatch:
    def __init__(self, db_path, n_features=N_FEATURES): 
        # Daha fazla özellik, daha hassas eşleşme demektir
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.db_path = db_path
        
        print(f"Veritabanı yükleniyor: {db_path} ...")
        self.db = np.load(db_path, allow_pickle=True)
        
        self.db_descriptors = self.db['descriptors']
        self.filenames = self.db['filenames']
        self.gps_data = self.db['gps_data']
        
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
        
        # FLANN Parametreleri
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,      
                            key_size=12,         
                            multi_probe_level=1) 
        
        self.search_params = dict(checks=30) 
        
        self.matcher = cv2.FlannBasedMatcher(index_params, self.search_params)
        self.matcher.add([self.all_descriptors])
        self.matcher.train()
        print(f"Model hazır! Toplam {len(self.all_descriptors)} özellik indekslendi.")
    # --- EKLENECEK YENİ METOT ---
    def detectAndCompute(self, frame, mask=None):
        """
        ROS Node tarafından çağrılan standart arayüz.
        Görüntüyü alır, (keypoints, descriptors) ikilisini döndürür.
        """
        # Breadcrumb kaydında boyut önemli değil ama tutarlılık için 
        # resize işlemi burada da yapılabilir veya ham haliyle işlenebilir.
        # Genelde breadcrumb için de resize (640px) önerilir.
        
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))
            
        return self.orb.detectAndCompute(frame, mask)
    def get_location(self, frame, min_match_count=4, top_k=5):
        """
        En iyi tek eşleşme yerine, en iyi 'top_k' eşleşmeyi liste olarak döndürür.
        Dönüş Formatı: [ (gps_data, filename, match_count), ... ]
        """
        # 1. Resize 
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))

        # 2. Özellik Çıkarımı
        kp, des = self.orb.detectAndCompute(frame, None)
        
        if des is None or len(des) < 5:
            return [] # Boş liste döndür

        # 3. Eşleştirme (KNN)
        matches = self.matcher.knnMatch(des, k=2)

        # 4. Ratio Test
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches_indices.append(m.trainIdx)

        if len(good_matches_indices) < min_match_count:
            return []

        # 5. Oylama
        matched_img_ids = self.desc_to_img_id[np.array(good_matches_indices)]
        
        # Hangi resim kaç kere eşleşti?
        counts = np.bincount(matched_img_ids)
        
        # En çok eşleşen top_k resmi bul
        # argsort küçükten büyüğe sıralar, [::-1] ile ters çevirip en büyükleri alırız
        best_indices = np.argsort(counts)[::-1]
        
        # Sadece 0'dan büyük skoru olanları ve en fazla top_k tanesini al
        candidates = []
        for idx in best_indices[:top_k]:
            score = counts[idx]
            if score < min_match_count:
                break # Skorlar sıralı olduğu için, eşik altındaysak sonrakiler de altındadır.
            
            candidates.append((self.gps_data[idx], self.filenames[idx], int(score)))

        # score üstünde olan candidate listelerini gönderiyoruz., candidate[][] şeklinde. max 5 tane döndürüyoruz.
        return candidates
    
    def match_frame_to_descriptors(self, frame, target_descriptors, min_match_count=8):
        """
        Gelişmiş Eşleştirme: Geri dönüş için 6 adet analiz verisi döndürür.
        """
        # Resize (Kayıtla aynı boyutta olmalı - 640px Genişlik)
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))
        else:
            scale = 1.0

        # Özellik Çıkar
        kp_curr, des_curr = self.orb.detectAndCompute(frame, None)
        
        if des_curr is None or len(des_curr) < 5 or target_descriptors is None:
            return None

        # Eşleştir
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # En iyi 50 eşleşme
        good_matches = matches[:50]

        if len(good_matches) < min_match_count:
            return None

        # --- GÖRSEL ANALİZ ---
        # Anlık Görüntüdeki Noktalar (Current)
        curr_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        
        # Merkez Hesabı
        centroid_x = np.mean(curr_pts[:, 0])
        centroid_y = np.mean(curr_pts[:, 1])
        
        # Merkezden Sapma (640px genişlik varsayımıyla merkez 320)
        # Eğer frame resize edilmediyse, shape'den alalım
        h, w = frame.shape[:2]
        center_x = w / 2
        center_y = h / 2
        
        error_x = centroid_x - center_x
        error_y = centroid_y - center_y 
        
        # Yayılım Analizi (Opsiyonel ama debug için iyi)
        spread_x = np.std(curr_pts[:, 0])

        # 6 DEĞER DÖNDÜRÜYORUZ
        return len(good_matches), error_x, error_y, spread_x, centroid_x, centroid_y
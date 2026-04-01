import cv2
import numpy as np
from skimage.feature import hog

# --- AYARLAR (Build scripti ile BİREBİR AYNI OLMALI) ---
TARGET_WIDTH = 480
TARGET_HEIGHT = 360
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16) # Yüksek Detay
HOG_CELLS_PER_BLOCK = (2, 2)

class HOGDetectAndMatch:
    def __init__(self, db_path):
        self.db_path = db_path
        print(f"[HOG-MAX] Veritabanı yükleniyor: {db_path} ...")
        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_features = self.db['hog_features'] 
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']
            # Hata kontrolü: Yüklenen veritabanının vektör boyutu, şu anki ayarlarla tutuyor mu?
            # (Basit bir kontrol, tam garanti değil ama fikir verir)
            expected_dim = self._get_expected_dim()
            loaded_dim = self.db_features.shape[1]
            if expected_dim != loaded_dim:
                 print(f"[HOG-MAX] UYARI: Veritabanı ayarları uyuşmuyor! Beklenen Vektör: {expected_dim}, Yüklenen: {loaded_dim}")
                 print("[HOG-MAX] Lütfen build_hog_database.py'yi bu ayarlarla tekrar çalıştırın.")

            print(f"[HOG-MAX] Hazır. Toplam {len(self.filenames)} resim. Vektör Boyutu: {loaded_dim}")
        except Exception as e:
            print(f"[HOG-MAX] Hata: {e}")
            self.db_features = None

    def _get_expected_dim(self):
        # Vektör boyutunu hesaplamak için dummy bir işlem
        dummy = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        feat = hog(dummy, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK, visualize=False, feature_vector=True)
        return len(feat)

    def detectAndCompute(self, frame, mask=None):
        # 1. Resize (640x480'e emin olalım)
        img_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 2. HOG Hesapla (Yüksek Detay)
        features = hog(img_resized, 
                       orientations=HOG_ORIENTATIONS, 
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, 
                       visualize=False, 
                       feature_vector=True)
        
        # Phase correlation için float32 resim ve HOG vektörü dön
        return np.float32(img_resized), features

    def get_location(self, frame, min_match_count=0, top_k=5):
        if self.db_features is None: return []
        
        _, query_feature = self.detectAndCompute(frame)
        if query_feature is None: return []

        distances = np.linalg.norm(self.db_features - query_feature, axis=1)
        best_indices = np.argsort(distances)[:top_k]
        
        candidates = []
        for idx in best_indices:
            dist = distances[idx]
            # Skorlama: Vektör boyutu büyüdüğü için mesafeler de büyür.
            # Normalizasyon katsayısını artırıyoruz (Deneme yanılma gerekebilir)
            # Maksimum beklenen mesafe ~30-40 olabilir.
            score = max(0, (40.0 - dist) * 2.5) 
            candidates.append((self.gps_data[idx], self.filenames[idx], score))

        return candidates

    def match_frame_to_descriptors(self, frame, target_data, min_match_count=0):
        # target_data = (ref_img_640x480_float, ref_hog_vec)
        ref_img, ref_vec = target_data
        curr_img, curr_vec = self.detectAndCompute(frame)
        
        # 1. Benzerlik Skoru (HOG)
        dist = np.linalg.norm(ref_vec - curr_vec)
        score = max(0, (40.0 - dist) * 2.5) 
        
        # 2. Konumsal Hata (Phase Correlation - 640x480 NATIVE)
        hann = cv2.createHanningWindow((TARGET_WIDTH, TARGET_HEIGHT), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(ref_img, curr_img, window=hann)
        
        dx, dy = shift
        
        # Ölçekleme: Görüntü zaten 640x480 olduğu için ölçek 1.0'dır.
        scale = 1.0
        
        # İşaret Yönü:
        # dx > 0 ise current image sağa kaymıştır -> Drone sola gitmiştir.
        # Hedef (Ref) sağda kalmıştır. Sağa dönmek için pozitif hata lazım.
        # Error = +dx (İşaretler deneme ile teyit edilmeli)
        error_x = dx * scale
        error_y = dy * scale
        
        # Merkez noktası artık 320, 240
        return score, error_x, error_y, 0.0, 320.0, 240.0
import cv2
import numpy as np
from skimage.feature import hog

# --- Parameters (must match the database-building scripts exactly) ---
TARGET_WIDTH = 480
TARGET_HEIGHT = 360
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)  # High detail
HOG_CELLS_PER_BLOCK = (2, 2)


class HOGDetectAndMatch:
    def __init__(self, db_path):
        self.db_path = db_path
        print(f"[HOG] Loading database: {db_path} ...")
        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_features = self.db['hog_features']
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']

            # Sanity check: verify the vector dimension matches current settings
            expected_dim = self._get_expected_dim()
            loaded_dim = self.db_features.shape[1]
            if expected_dim != loaded_dim:
                print(f"[HOG] WARNING: Database dimension mismatch! "
                      f"Expected {expected_dim}, loaded {loaded_dim}. "
                      f"Rebuild the database with the current settings.")

            print(f"[HOG] Ready. {len(self.filenames)} images. Vector dim: {loaded_dim}")
        except Exception as e:
            print(f"[HOG] Database load error: {e}")
            self.db_features = None

    def _get_expected_dim(self):
        """Compute the expected HOG vector length for current settings."""
        dummy = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        feat = hog(dummy, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK, visualize=False, feature_vector=True)
        return len(feat)

    def detectAndCompute(self, frame, mask=None):
        """
        Compute HOG features for a frame.
        Returns (resized_gray_float32, hog_vector) — used for both breadcrumbs
        and phase-correlation backtracking.
        """
        # 1. Resize to processing resolution
        img_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # 2. Compute HOG descriptor
        features = hog(img_resized,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       visualize=False,
                       feature_vector=True)

        # Return float32 image for phase correlation alongside the HOG vector
        return np.float32(img_resized), features

    def get_location(self, frame, min_match_count=0, top_k=5):
        """Find the best matching database positions via HOG distance."""
        if self.db_features is None:
            return []

        _, query_feature = self.detectAndCompute(frame)
        if query_feature is None:
            return []

        distances = np.linalg.norm(self.db_features - query_feature, axis=1)
        best_indices = np.argsort(distances)[:top_k]

        candidates = []
        for idx in best_indices:
            dist = distances[idx]
            # Scoring: higher-dimensional vectors produce larger raw distances.
            # The normalization factor below was tuned empirically (max expected ~30-40).
            score = max(0, (40.0 - dist) * 2.5)
            candidates.append((self.gps_data[idx], self.filenames[idx], score))

        return candidates

    def match_frame_to_descriptors(self, frame, target_data, min_match_count=0):
        """
        Backtracking match using HOG similarity + phase correlation for positional error.
        target_data: (ref_img_float32, ref_hog_vector)
        """
        ref_img, ref_vec = target_data
        curr_img, curr_vec = self.detectAndCompute(frame)

        # 1. HOG similarity score
        dist = np.linalg.norm(ref_vec - curr_vec)
        score = max(0, (40.0 - dist) * 2.5)

        # 2. Positional error via phase correlation (native 480x360 resolution)
        hann = cv2.createHanningWindow((TARGET_WIDTH, TARGET_HEIGHT), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(ref_img, curr_img, window=hann)
        dx, dy = shift

        # Sign convention:
        # dx > 0 means the current frame shifted right relative to the reference
        # → the drone moved left → positive error drives correction to the right
        # (Verify sign direction experimentally on the actual vehicle)
        error_x = dx
        error_y = dy

        # Image center is (240, 180) for 480x360 resolution
        return score, error_x, error_y, 0.0, TARGET_WIDTH / 2, TARGET_HEIGHT / 2
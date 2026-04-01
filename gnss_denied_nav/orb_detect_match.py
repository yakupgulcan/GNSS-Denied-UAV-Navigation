import cv2
import numpy as np
import os


# Default number of ORB features (overridden via db_path parameter)
N_FEATURES = 2000

class ORBDetectAndMatch:
    def __init__(self, db_path, n_features=N_FEATURES):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.db_path = db_path

        print(f"[ORB] Loading database: {db_path} ...")
        self.db = np.load(db_path, allow_pickle=True)

        self.db_descriptors = self.db['descriptors']
        self.filenames = self.db['filenames']
        self.gps_data = self.db['gps_data']

        # --- Flatten database descriptors ---
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

        # FLANN parameters for binary (ORB) descriptors
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)

        self.search_params = dict(checks=30)

        self.matcher = cv2.FlannBasedMatcher(index_params, self.search_params)
        self.matcher.add([self.all_descriptors])
        self.matcher.train()
        print(f"[ORB] Ready. {len(self.all_descriptors)} features indexed.")
    def detectAndCompute(self, frame, mask=None):
        """Standard interface called by the visual estimator node."""
        # Resize to 640px for consistency with breadcrumb recordings.
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))
        return self.orb.detectAndCompute(frame, mask)

    def get_location(self, frame, min_match_count=4, top_k=5):
        """
        Returns the top-k best matching database images as a list.
        Format: [(gps_data, filename, match_count), ...]
        """
        # 1. Resize
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))

        # 2. Feature extraction
        kp, des = self.orb.detectAndCompute(frame, None)

        if des is None or len(des) < 5:
            return []

        # 3. KNN matching
        matches = self.matcher.knnMatch(des, k=2)

        # 4. Ratio test
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches_indices.append(m.trainIdx)

        if len(good_matches_indices) < min_match_count:
            return []

        # 5. Voting: count matches per image
        matched_img_ids = self.desc_to_img_id[np.array(good_matches_indices)]
        counts = np.bincount(matched_img_ids)

        # Sort descending; argsort is ascending so reverse it
        best_indices = np.argsort(counts)[::-1]

        # Return only candidates above the score threshold (max top_k)
        candidates = []
        for idx in best_indices[:top_k]:
            score = counts[idx]
            if score < min_match_count:
                break  # Scores are sorted; if below threshold, rest are too
            candidates.append((self.gps_data[idx], self.filenames[idx], int(score)))

        return candidates
    
    def match_frame_to_descriptors(self, frame, target_descriptors, min_match_count=8):
        """Backtracking match: returns 6 analysis values for alignment."""
        # Resize to match recorded breadcrumb size (640px width)
        height, width = frame.shape[:2]
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (target_width, new_height))
        else:
            scale = 1.0

        # Extract features
        kp_curr, des_curr = self.orb.detectAndCompute(frame, None)

        if des_curr is None or len(des_curr) < 5 or target_descriptors is None:
            return None

        # Brute-force match (Hamming distance for binary ORB descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Take top 50 matches
        good_matches = matches[:50]

        if len(good_matches) < min_match_count:
            return None

        # Compute centroid and error from image center
        curr_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        centroid_x = np.mean(curr_pts[:, 0])
        centroid_y = np.mean(curr_pts[:, 1])

        h, w = frame.shape[:2]
        center_x = w / 2
        center_y = h / 2

        error_x = centroid_x - center_x
        error_y = centroid_y - center_y

        # Spread: optional but useful for debugging match quality
        spread_x = np.std(curr_pts[:, 0])

        return len(good_matches), error_x, error_y, spread_x, centroid_x, centroid_y
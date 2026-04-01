import cv2
import numpy as np
import os

TARGET_WIDTH = 640


class AKAZEDetectAndMatch:
    def __init__(self, db_path, threshold=0.001):
        # Use the same AKAZE parameters as the database-building scripts
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            threshold=threshold,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.db_path = db_path
        print(f"[AKAZE] Loading database: {db_path} ...")

        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_descriptors = self.db['descriptors']
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']
        except Exception as e:
            print(f"[AKAZE] Database load error: {e}")
            return

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

        # FLANN LSH for binary descriptors; increased checks for better accuracy
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.matcher.add([self.all_descriptors])
        self.matcher.train()
        print(f"[AKAZE] Ready. {len(self.all_descriptors)} features indexed.")

    def _preprocess(self, frame):
        """Resize to TARGET_WIDTH and apply CLAHE contrast enhancement."""
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
        return self.clahe.apply(frame)

    def get_location(self, frame, min_match_count=8, top_k=5):
        """Scan the feature database to find the best matching position."""
        frame_proc = self._preprocess(frame)
        kp, des = self.akaze.detectAndCompute(frame_proc, None)

        if des is None or len(des) < 5:
            return []

        matches = self.matcher.knnMatch(des, k=2)
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            # 0.75 is a good ratio test threshold for AKAZE
            if m.distance < 0.75 * n.distance:
                good_matches_indices.append(m.trainIdx)

        if len(good_matches_indices) < min_match_count:
            return []

        matched_img_ids = self.desc_to_img_id[np.array(good_matches_indices)]
        counts = np.bincount(matched_img_ids)
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
        Backtracking match with statistical outlier rejection.
        """
        # 1. Preprocessing (resize + CLAHE)
        frame_proc = self._preprocess(frame)
        h, w = frame_proc.shape[:2]
        center_x, center_y = w / 2, h / 2

        # 2. Feature extraction
        kp_curr, des_curr = self.akaze.detectAndCompute(frame_proc, None)
        if des_curr is None or len(des_curr) < 5 or target_descriptors is None:
            return None

        # 3. Brute-force matching (Hamming distance for binary AKAZE descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)

        # Initial filter: sort by distance, keep top 30% or max 50
        matches = sorted(matches, key=lambda x: x.distance)
        top_n = min(50, int(len(matches) * 0.3))
        good_matches = matches[:top_n]

        if len(good_matches) < min_match_count:
            return None

        # 4. Statistical outlier rejection (Z-score filtering)
        # Full homography is not available (we only have descriptors, not keypoints
        # from the database), so we use robust statistics instead.
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        mean = np.mean(src_pts, axis=0)
        std = np.std(src_pts, axis=0)

        # Reject points beyond 2 standard deviations from the centroid
        filtered_pts = [
            pt for pt in src_pts
            if abs(pt[0] - mean[0]) < 2 * std[0] and abs(pt[1] - mean[1]) < 2 * std[1]
        ]

        if len(filtered_pts) < min_match_count:
            return None

        filtered_pts = np.array(filtered_pts)

        # 5. Compute centroid and error from filtered inliers
        centroid_x = np.mean(filtered_pts[:, 0])
        centroid_y = np.mean(filtered_pts[:, 1])

        error_x = centroid_x - center_x
        error_y = centroid_y - center_y
        spread_x = np.std(filtered_pts[:, 0])

        return len(filtered_pts), error_x, error_y, spread_x, centroid_x, centroid_y
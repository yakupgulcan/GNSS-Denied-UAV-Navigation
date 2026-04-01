import cv2
import numpy as np
import os

# Target resize width (reduce if performance is an issue)
TARGET_WIDTH = 640


class BRISKDetectAndMatch:
    def __init__(self, db_path, thresh=60):
        # octaves=0 disables scale invariance but maximizes speed
        self.brisk = cv2.BRISK_create(thresh=thresh, octaves=0)
        self.db_path = db_path

        print(f"[BRISK] Loading database: {db_path} ...")

        try:
            self.db = np.load(db_path, allow_pickle=True)
            self.db_descriptors = self.db['descriptors']
            self.filenames = self.db['filenames']
            self.gps_data = self.db['gps_data']
        except Exception as e:
            print(f"[BRISK] Database load error: {e}")
            # Continue with empty lists to prevent crash
            self.db_descriptors = []
            self.filenames = []
            self.gps_data = []

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

            # FLANN LSH (fastest for binary descriptors)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=30)

            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.matcher.add([self.all_descriptors])
            self.matcher.train()
            print(f"[BRISK] Ready. {len(self.all_descriptors)} features indexed.")
        else:
            print("[BRISK] WARNING: No valid features found in database!")
            self.matcher = None

    def _resize_frame(self, frame):
        """Resize frame to TARGET_WIDTH while preserving aspect ratio."""
        height, width = frame.shape[:2]
        if width > TARGET_WIDTH:
            scale = TARGET_WIDTH / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))
        return frame

    def detectAndCompute(self, frame, mask=None):
        """
        Standard interface for the visual estimator node.
        Used when recording breadcrumb keyframes.
        Resize for consistency so stored and query features are at the same scale.
        """
        frame_resized = self._resize_frame(frame)
        return self.brisk.detectAndCompute(frame_resized, mask)

    def get_location(self, frame, min_match_count=8, top_k=5):
        """Scan the feature database to find the best matching position."""
        if self.matcher is None:
            return []

        frame = self._resize_frame(frame)
        kp, des = self.brisk.detectAndCompute(frame, None)

        if des is None or len(des) < 5:
            return []

        matches = self.matcher.knnMatch(des, k=2)
        good_matches_indices = []
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            # 0.75 is a good ratio test threshold for BRISK
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
        """Backtracking match against a breadcrumb descriptor."""
        frame = self._resize_frame(frame)
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2

        kp_curr, des_curr = self.brisk.detectAndCompute(frame, None)
        if des_curr is None or len(des_curr) < 5 or target_descriptors is None:
            return None

        # Hamming distance for binary BRISK descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_curr, target_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Take top 50 matches
        good_matches = matches[:50]

        if len(good_matches) < min_match_count:
            return None

        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        centroid_x = np.mean(src_pts[:, 0])
        centroid_y = np.mean(src_pts[:, 1])

        error_x = centroid_x - center_x
        error_y = centroid_y - center_y
        spread_x = np.std(src_pts[:, 0])

        return len(good_matches), error_x, error_y, spread_x, centroid_x, centroid_y
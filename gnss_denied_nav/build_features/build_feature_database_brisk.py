import cv2
import numpy as np
import os
import json

# --- PERFORMANCE SETTINGS ---
frames_path = "~/frames_2026-01-09_08-47-17"
BRISK_THRESH = 60    # Increased threshold (fewer but stronger points)
BRISK_OCTAVES = 0    # 0 = Only scan original size (VERY FAST)
MAX_FEATURES = 1000  

def build_brisk_database(frames_dir):
    # Setting Octaves=0 turns off the pyramid processing. This makes it as fast as ORB.
    brisk = cv2.BRISK_create(thresh=BRISK_THRESH, octaves=BRISK_OCTAVES)

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = []

    json_path = os.path.join(frames_dir, "frames_details.json")
    if not os.path.exists(json_path): return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"Building FAST BRISK Database (Octaves={BRISK_OCTAVES})...")

    count = 0
    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Feature Extraction
        kp, des = brisk.detectAndCompute(img, None)
        
        if des is None or len(kp) == 0: continue

        # Select the best 1000 features
        if len(kp) > MAX_FEATURES:
            kp_des_pair = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)[:MAX_FEATURES]
            kp = [x[0] for x in kp_des_pair]
            des = np.array([x[1] for x in kp_des_pair])

        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        gps_data.append(entry.get("gps", {}))
        headings_db.append(entry.get("heading", 0.0))
        
        count += 1
        if count % 100 == 0: print(f"{count} frames processed...")

    output_path = os.path.join(frames_dir, "features_db_brisk_60_0.npz")
    
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32)
    )
    print(f"Save Completed: {output_path}")

if __name__ == "__main__":
    real_frames_dir = os.path.expanduser(frames_path)
    build_brisk_database(real_frames_dir)
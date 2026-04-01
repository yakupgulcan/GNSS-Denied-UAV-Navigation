import cv2
import numpy as np
import os
import json

# --- SETTINGS ---
frames_path = "~/frames_2026-01-17_17-22-56"
N_FEATURES = 2000 # Good starting point for SIFT

def build_sift_database(frames_dir):
    """
    Builds the SIFT feature database.
    Saved data: descriptors, keypoints, filenames, gps_data, headings
    """

    # 1. Initialize SIFT
    # Available natively in modern OpenCV (4.4+) since the patent expired.
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = []

    json_path = os.path.join(frames_dir, "frames_details.json")
    
    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"Building database using SIFT. Total {len(frames_info)} frames...")

    count = 0
    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        
        if not os.path.exists(img_path):
            continue

        # Read image (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 2. SIFT Feature Extraction
        kp, des = sift.detectAndCompute(img, None)
        
        # Skip if no features found
        if des is None or len(kp) == 0:
            continue

        # Serialize KeyPoints (x, y, size, angle)
        # SIFT keypoint structure is compatible with ORB, we can use the same logic.
        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        
        gps_data.append(entry.get("gps", {}))
        headings_db.append(entry.get("heading", 0.0))
        
        count += 1
        if count % 50 == 0:
            print(f"{count} frames processed...")

    # Save as .npz
    output_path = os.path.join(frames_dir, f"features_db_sift_{N_FEATURES}.npz")
    
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32)
    )

    print("-" * 30)
    print(f"Process completed.")
    print(f"Images processed: {len(filenames)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    real_frames_dir = os.path.expanduser(frames_path)
    build_sift_database(real_frames_dir)
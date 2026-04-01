import cv2
import numpy as np
import os
import json
"""
Reads the collected images and their metadata json file,
extracts ORB features, and saves them to an npz file.
"""
frames_path = "~/frames_2026-01-09_08-47-17"

def build_feature_database(frames_dir):
    """
    Builds the ORB feature database.
    Saved data: descriptors, keypoints, filenames, gps_data, headings
    """

    # 1. nfeatures set to 2000 (Faster but less dense detail)
    orb = cv2.ORB_create(nfeatures=2000)

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = [] # Header list

    json_path = os.path.join(frames_dir, "frames_details.json")
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"Processing total {len(frames_info)} frames...")

    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        
        # Check if file exists
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Feature extraction
        kp, des = orb.detectAndCompute(img, None)
        
        # Skip this frame if no features found
        if des is None or len(kp) == 0:
            continue

        # Serialize KeyPoints to a storable format (x, y, size, angle)
        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        
        # Get GPS data
        gps_data.append(entry.get("gps", {}))
        
        # Get heading data (assume 0.0 if not present)
        headings_db.append(entry.get("heading", 0.0))

    # Save as .npz
    output_path = os.path.join(frames_dir, "features_db_2000.npz")
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32) 
    )

    print(f"Process completed. Database created for {len(filenames)} images.")
    print(f"File: {output_path}")

if __name__ == "__main__":
    frames_dir = os.path.expanduser(frames_path)
    build_feature_database(frames_dir)
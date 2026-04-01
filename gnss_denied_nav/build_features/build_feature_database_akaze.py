import cv2
import numpy as np
import os
import json

# --- ADVANCED SETTINGS ---
frames_path = "~/frames_2026-01-05_01-06-36"
# Threshold increased (Reduces noise, increases quality)
AKAZE_THRESH = 0.001 
MAX_FEATURES = 1500  # Maximum number of features to save

def build_akaze_database(frames_dir):
    """
    Builds the AKAZE feature database (with CLAHE support).
    """
    # AKAZE Parameters (More precise settings)
    # diffusivity=cv2.KAZE_DIFF_PM_G2 -> Better edge preservation
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=AKAZE_THRESH,
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=cv2.KAZE_DIFF_PM_G2
    )

    # Contrast Enhancer (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    descriptors_db = []
    keypoints_db = []
    filenames = []
    gps_data = []
    headings_db = []

    json_path = os.path.join(frames_dir, "frames_details.json")
    if not os.path.exists(json_path): return

    with open(json_path) as f:
        frames_info = json.load(f)

    print(f"Building database using Advanced AKAZE...")

    count = 0
    for entry in frames_info:
        img_path = os.path.join(frames_dir, entry["filename"])
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # --- IMPROVEMENT 1: Apply CLAHE ---
        # Exposes details in the image, allowing AKAZE to find better keypoints.
        img_enhanced = clahe.apply(img)

        # AKAZE Feature Extraction
        kp, des = akaze.detectAndCompute(img_enhanced, None)
        
        if des is None or len(kp) < 5: continue
        
        # --- IMPROVEMENT 2: Select Best Keypoints ---
        if len(kp) > MAX_FEATURES:
            # Sort by response value
            kp_des_pair = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)[:MAX_FEATURES]
            kp = [x[0] for x in kp_des_pair]
            des = np.array([x[1] for x in kp_des_pair])

        # Serialize Keypoints
        kp_simple = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp], dtype=np.float32)

        descriptors_db.append(des)
        keypoints_db.append(kp_simple)
        filenames.append(entry["filename"])
        gps_data.append(entry.get("gps", {}))
        headings_db.append(entry.get("heading", 0.0))
        
        count += 1
        if count % 50 == 0: print(f"{count} frames processed...")

    # Save
    output_path = os.path.join(frames_dir, "features_db_akaze.npz")
    np.savez_compressed(
        output_path,
        descriptors=np.array(descriptors_db, dtype=object),
        keypoints=np.array(keypoints_db, dtype=object),
        filenames=np.array(filenames, dtype=object),
        gps_data=np.array(gps_data, dtype=object),
        headings=np.array(headings_db, dtype=np.float32)
    )
    print(f"Completed: {output_path}")

if __name__ == "__main__":
    real_frames_dir = os.path.expanduser(frames_path)
    build_akaze_database(real_frames_dir)
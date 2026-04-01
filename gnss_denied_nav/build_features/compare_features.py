#!/usr/bin/env python3
"""
This script visually compares SIFT and ORB Keypoints side-by-side on an image.
Users can provide any image path to run the test.
"""

import cv2
import numpy as np
import argparse

def draw_keypoints_side_by_side(image_path):
    # Read Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Image could not be read -> {image_path}")
        return

    # Resize to a manageable size (scale down if too large)
    MAX_WIDTH = 1200
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * scale)))

    # --- 1. SIFT ---
    # Set nfeatures=1000 for a fair comparison against ORB
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_sift, _ = sift.detectAndCompute(img, None)
    
    # SIFT Image (Red Points)
    # cv2.drawKeypoints draws the points.
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS -> Shows size and orientation (circles)
    img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Write Text
    cv2.putText(img_sift, f"SIFT: {len(kp_sift)} features", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 2. ORB ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp_orb, _ = orb.detectAndCompute(img, None)
    
    # ORB Image (Green Points)
    img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0), flags=0) # flags=0 just points
    
    # Write Text
    cv2.putText(img_orb, f"ORB: {len(kp_orb)} features", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 3. COMBINE AND SHOW ---
    # Put side-by-side
    combined = np.hstack((img_sift, img_orb))
    
    # Save to file
    output_file = "comparison_sift_vs_orb.jpg"
    cv2.imwrite(output_file, combined)
    print(f"Comparison saved as: {output_file}")
    print(f"SIFT Feature Count: {len(kp_sift)}")
    print(f"ORB Feature Count : {len(kp_orb)}")

if __name__ == "__main__":
    # Provides default behavior or explicit argument
    # Example usage: python3 compare_features.py --image /path/to/image.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    draw_keypoints_side_by_side(args.image)

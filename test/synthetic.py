"""Synthetic nadir-camera world shared by the tests (no dataset or ROS needed)."""
import math

import cv2
import numpy as np

from gnss_denied_nav.localization import CameraModel

ORIGIN = (-35.3766471, 149.1652375)
ALT = 30.0
CAM = CameraModel(640, 480, 2.0)
GROUND_RES = 0.05  # m per ground-texture pixel


def make_ground(seed=0, size=4000):
    """Unique, feature-rich ground texture (no repeating tiles)."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (size // 16, size // 16), dtype=np.uint8),
                     (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(1500):
        x, y = rng.integers(0, size, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(5, 30)), int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def render(ground, east, north, heading_deg, alt=ALT):
    """Nadir camera image at ENU (east, north), image-up = heading."""
    s = CAM.gsd(alt) / GROUND_RES                          # ground px per image px
    c = ground.shape[0] / 2.0
    gx, gy = c + east / GROUND_RES, c - north / GROUND_RES  # ground image: x east, y south
    h = math.radians(heading_deg)
    fwd = np.array([math.sin(h), -math.cos(h)])            # ground-px direction of image-up
    right = np.array([math.cos(h), math.sin(h)])
    cx, cy = CAM.center
    M = np.column_stack([right * s, -fwd * s])
    t = np.array([gx, gy]) - M @ np.array([cx, cy])
    A = np.hstack([M, t[:, None]])
    return cv2.warpAffine(ground, A, (CAM.width, CAM.height),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

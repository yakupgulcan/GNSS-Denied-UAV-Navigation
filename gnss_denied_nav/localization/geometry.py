"""Homography verification and homography -> metric ENU pose.

The camera looks straight down over roughly flat terrain, so a query image
and an overlapping map image are related by a homography. Projecting the
query image center into the map image gives a pixel offset from the map
frame's center; scaled by the map frame's ground sampling distance and
rotated by its heading, that offset is the query's ENU position relative to
the map frame's GPS tag.

Image axes: u to the right, v down. With the camera pointing down and its
top edge facing the vehicle's nose, image-up is "forward" and image-right is
"right" in the body frame. Verified against the survey dataset's GPS tags.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VerifyConfig:
    ransac_px: float = 3.0
    min_inliers: int = 15
    min_inlier_ratio: float = 0.15
    max_scale_change: float = 2.0     # allowed query/ref altitude ratio
    max_perspective: float = 0.3      # |h20*w| + |h21*h| (nadir => ~0)
    sigma_floor_m: float = 1.0


@dataclass
class Homography:
    H: np.ndarray                     # query pixels -> ref pixels
    n_inliers: int
    inlier_ratio: float
    rms_px: float


def estimate_homography(corr, cfg):
    """RANSAC homography from correspondences, or (None, reason)."""
    n = len(corr)
    if n < max(4, cfg.min_inliers):
        return None, 'few_matches'
    H, mask = cv2.findHomography(corr.pts_query, corr.pts_ref, cv2.USAC_MAGSAC, cfg.ransac_px)
    if H is None or mask is None:
        return None, 'no_homography'
    inl = mask.ravel().astype(bool)
    n_inl = int(inl.sum())
    if n_inl < cfg.min_inliers:
        return None, 'few_inliers'
    ratio = n_inl / n
    if ratio < cfg.min_inlier_ratio:
        return None, 'low_inlier_ratio'
    proj = cv2.perspectiveTransform(corr.pts_query[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - corr.pts_ref[inl]) ** 2, axis=1))))
    return Homography(H, n_inl, ratio, rms), None


def homography_to_pose(hom, camera, ref_enu, ref_heading_deg, ref_alt, cfg):
    """Convert a verified homography into (x, y, heading, sigma, scale) or (None, reason).

    Returns a dict on success.
    """
    H = hom.H / hom.H[2, 2]
    A = H[:2, :2]
    det = float(np.linalg.det(A))
    if det <= 0:
        return None, 'reflection'
    scale = math.sqrt(det)
    if not (1.0 / cfg.max_scale_change <= scale <= cfg.max_scale_change):
        return None, 'scale'
    if abs(H[2, 0]) * camera.width + abs(H[2, 1]) * camera.height > cfg.max_perspective:
        return None, 'perspective'

    cx, cy = camera.center
    u, v = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0, 0]
    # The query center must land on (or near) the map image to be trusted.
    if not (-0.5 * camera.width <= u <= 1.5 * camera.width
            and -0.5 * camera.height <= v <= 1.5 * camera.height):
        return None, 'off_image'

    gsd = camera.gsd(ref_alt)
    forward = -(v - cy) * gsd
    right = (u - cx) * gsd
    h = math.radians(ref_heading_deg)
    east = forward * math.sin(h) + right * math.cos(h)
    north = forward * math.cos(h) - right * math.sin(h)

    rot_deg = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    heading = (ref_heading_deg + rot_deg) % 360.0

    sigma = (cfg.sigma_floor_m * math.sqrt(cfg.min_inliers / hom.n_inliers)
             + gsd * hom.rms_px)
    return {
        'x': ref_enu[0] + east,
        'y': ref_enu[1] + north,
        'heading_deg': heading,
        'sigma_m': sigma,
        'scale': scale,
    }, None

"""Plain data types shared by the localization pipeline."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Features:
    """Local features of one image.

    Keypoints are pixel coordinates in the original (un-resized) image.
    """

    kpts: np.ndarray            # (N, 2) float32
    desc: np.ndarray            # (N, D) float32 or uint8
    image_size: tuple           # (width, height)

    def __len__(self):
        return len(self.kpts)


@dataclass
class Correspondences:
    """Matched pixel pairs between a query and a reference image."""

    pts_query: np.ndarray       # (M, 2) float32
    pts_ref: np.ndarray         # (M, 2) float32

    def __len__(self):
        return len(self.pts_query)


@dataclass
class LocalizationResult:
    """Metric position of the query image, from one verified map frame."""

    x: float                    # ENU east (m)
    y: float                    # ENU north (m)
    heading_deg: float          # camera heading, compass convention (deg)
    sigma_m: float              # 1-sigma horizontal uncertainty (m)
    n_inliers: int
    inlier_ratio: float
    scale: float                # query/ref ground-sampling ratio (~ alt ratio)
    ref_index: int              # map frame that produced the fix
    candidates: list = field(default_factory=list)  # map frames tried

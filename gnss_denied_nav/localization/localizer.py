"""Localizer: retrieve -> match -> verify -> metric pose."""
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from gnss_denied_nav.localization.camera import CameraModel
from gnss_denied_nav.localization.features import create_extractor
from gnss_denied_nav.localization.geometry import (
    VerifyConfig, estimate_homography, homography_to_pose)
from gnss_denied_nav.localization.matchers import create_matcher
from gnss_denied_nav.localization.retrieval import VotingRetriever
from gnss_denied_nav.localization.types import Correspondences, LocalizationResult

_ROTATE = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def heading_diff(a, b):
    """Signed difference a - b in degrees, wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def rotation_steps(query_heading, ref_heading):
    """Clockwise 90-degree turns that bring the query image to the reference heading."""
    return int(round((query_heading - ref_heading) % 360.0 / 90.0)) % 4


def unrotate_points(pts, k, width, height):
    """Map pixels of the query image turned k x 90 deg clockwise back to the original image."""
    if k == 0 or len(pts) == 0:
        return pts
    u, v = pts[:, 0], pts[:, 1]
    if k == 1:
        x, y = v, height - 1 - u
    elif k == 2:
        x, y = width - 1 - u, height - 1 - v
    else:
        x, y = width - 1 - v, u
    return np.column_stack([x, y]).astype(np.float32)


@dataclass
class LocalizerConfig:
    max_candidates: int = 5          # map frames matched per query
    match_ratio: float = 0.8         # Lowe ratio for pair-wise matching
    global_retrieval: bool = True    # build the voting index (needed without a prior)
    min_votes: int = 10
    # Reject a fix whose heading differs from the given heading prior by more
    # than this (deg). None disables the check.
    max_heading_error_deg: float = 30.0
    # Stop trying further candidates (nearest first) once a fix has at least
    # this many inliers. None tries all candidates and keeps the best.
    accept_inliers: int = None
    verify: VerifyConfig = field(default_factory=VerifyConfig)


class Localizer:

    def __init__(self, db, camera=None, config=None):
        self.db = db
        self.camera = camera or CameraModel(*db.image_size)
        self.cfg = config or LocalizerConfig()
        if db.enu is None:
            raise ValueError('MapDatabase.set_origin() must be called first')
        # Query features must be extracted exactly like the map's.
        self.extractor = create_extractor(db.method, **db.params)
        self.matcher = create_matcher(self.extractor, ratio=self.cfg.match_ratio)
        # Global retrieval votes with descriptors; image-based maps have none.
        self.retriever = (VotingRetriever(db, self.extractor.binary)
                          if self.cfg.global_retrieval and not db.image_based else None)

    def localize(self, gray, prior=None, radius=None, exclude=(), heading=None):
        """Localize one grayscale image.

        prior/radius: predicted (x, y) and search radius in meters. Without a
        prior the global voting retriever is used.
        exclude: map frame indices to skip (leave-one-out evaluation).
        heading: approximate camera heading (compass deg), optional. Used to
        turn the query image by multiples of 90 deg towards each map frame
        (for features that are not rotation invariant, e.g. SuperPoint) and
        to reject fixes with an inconsistent heading.

        Returns (LocalizationResult or None, info dict).
        """
        t0 = time.perf_counter()
        query = self.extractor.extract(gray)
        t_extract = time.perf_counter()
        upright = heading is not None and not getattr(self.extractor, 'rotation_invariant', True)
        turned = {0: query}
        ms_turn = 0.0

        if prior is not None:
            candidates = list(self.db.frames_near(prior[0], prior[1], radius,
                                                  self.cfg.max_candidates, exclude))
        elif self.retriever is not None:
            candidates = [i for i, _ in self.retriever.retrieve(
                query, self.cfg.max_candidates, self.cfg.min_votes, exclude)]
        else:
            candidates = []
        t_retrieve = time.perf_counter()

        best, rejects, turns = None, Counter(), []
        h, w = gray.shape[:2]
        for i in candidates:
            k = rotation_steps(heading, self.db.heading[i]) if upright else 0
            turns.append(k)
            if k not in turned:
                t = time.perf_counter()
                turned[k] = self.extractor.extract(cv2.rotate(gray, _ROTATE[k]))
                ms_turn += (time.perf_counter() - t) * 1e3
            corr = self.matcher.match(turned[k], self.db.features(i))
            if k:
                corr = Correspondences(unrotate_points(corr.pts_query, k, w, h), corr.pts_ref)
            hom, reason = estimate_homography(corr, self.cfg.verify)
            if hom is None:
                rejects[reason] += 1
                continue
            pose, reason = homography_to_pose(
                hom, self.camera, self.db.enu[i], self.db.heading[i], self.db.rel_alt[i],
                self.cfg.verify)
            if pose is None:
                rejects[reason] += 1
                continue
            if (heading is not None and self.cfg.max_heading_error_deg is not None
                    and abs(heading_diff(pose['heading_deg'], heading)) > self.cfg.max_heading_error_deg):
                rejects['heading'] += 1
                continue
            if best is None or hom.n_inliers > best.n_inliers:
                best = LocalizationResult(n_inliers=hom.n_inliers, inlier_ratio=hom.inlier_ratio,
                                          ref_index=int(i), candidates=candidates, **pose)
            if self.cfg.accept_inliers is not None and best.n_inliers >= self.cfg.accept_inliers:
                break
        t_end = time.perf_counter()

        info = {
            'n_features': len(query),
            'candidates': candidates,
            'rejects': dict(rejects),
            'rotations': turns,
            'n_tried': len(turns),
            'ms_extract': (t_extract - t0) * 1e3 + ms_turn,
            'ms_retrieve': (t_retrieve - t_extract) * 1e3,
            'ms_match': (t_end - t_retrieve) * 1e3 - ms_turn,
            'ms_total': (t_end - t0) * 1e3,
        }
        return best, info

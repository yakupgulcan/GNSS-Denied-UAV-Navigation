"""Localizer: retrieve -> match -> verify -> metric pose."""
import time
from collections import Counter
from dataclasses import dataclass, field

from gnss_denied_nav.localization.camera import CameraModel
from gnss_denied_nav.localization.features import create_extractor
from gnss_denied_nav.localization.geometry import (
    VerifyConfig, estimate_homography, homography_to_pose)
from gnss_denied_nav.localization.matchers import create_matcher
from gnss_denied_nav.localization.retrieval import VotingRetriever
from gnss_denied_nav.localization.types import LocalizationResult


@dataclass
class LocalizerConfig:
    max_candidates: int = 5          # map frames matched per query
    match_ratio: float = 0.8         # Lowe ratio for pair-wise matching
    global_retrieval: bool = True    # build the voting index (needed without a prior)
    min_votes: int = 10
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
        self.retriever = (VotingRetriever(db, self.extractor.binary)
                          if self.cfg.global_retrieval else None)

    def localize(self, gray, prior=None, radius=None, exclude=()):
        """Localize one grayscale image.

        prior/radius: predicted (x, y) and search radius in meters. Without a
        prior the global voting retriever is used.
        exclude: map frame indices to skip (leave-one-out evaluation).

        Returns (LocalizationResult or None, info dict).
        """
        t0 = time.perf_counter()
        query = self.extractor.extract(gray)
        t_extract = time.perf_counter()

        if prior is not None:
            candidates = list(self.db.frames_near(prior[0], prior[1], radius,
                                                  self.cfg.max_candidates, exclude))
        elif self.retriever is not None:
            candidates = [i for i, _ in self.retriever.retrieve(
                query, self.cfg.max_candidates, self.cfg.min_votes, exclude)]
        else:
            candidates = []
        t_retrieve = time.perf_counter()

        best, rejects = None, Counter()
        for i in candidates:
            corr = self.matcher.match(query, self.db.features(i))
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
            if best is None or hom.n_inliers > best.n_inliers:
                best = LocalizationResult(n_inliers=hom.n_inliers, inlier_ratio=hom.inlier_ratio,
                                          ref_index=int(i), candidates=candidates, **pose)
        t_end = time.perf_counter()

        info = {
            'n_features': len(query),
            'candidates': candidates,
            'rejects': dict(rejects),
            'ms_extract': (t_extract - t0) * 1e3,
            'ms_retrieve': (t_retrieve - t_extract) * 1e3,
            'ms_match': (t_end - t_retrieve) * 1e3,
            'ms_total': (t_end - t0) * 1e3,
        }
        return best, info

"""Pair-wise matchers: query Features x reference Features -> Correspondences.

A matcher only has to implement ``match(query, ref) -> Correspondences``.
Learned matchers (LightGlue, or a detector-free one like LoFTR that works on
images instead of Features) plug in behind the same call.
"""
import cv2
import numpy as np

from gnss_denied_nav.localization.types import Correspondences

_EMPTY = Correspondences(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))


class DescriptorMatcher:
    """Brute-force k-NN matching with Lowe's ratio test."""

    def __init__(self, binary, ratio=0.8):
        self.norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
        self.ratio = ratio
        self.bf = cv2.BFMatcher(self.norm)

    def match(self, query, ref):
        if len(query) < 2 or len(ref) < 2:
            return _EMPTY
        pairs = self.bf.knnMatch(query.desc, ref.desc, k=2)
        idx_q, idx_r = [], []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                idx_q.append(pair[0].queryIdx)
                idx_r.append(pair[0].trainIdx)
        if not idx_q:
            return _EMPTY
        return Correspondences(query.kpts[idx_q], ref.kpts[idx_r])


def create_matcher(extractor, ratio=0.8):
    """Matcher for the extractor's features (ratio only applies to descriptor matching)."""
    if getattr(extractor, 'matcher', None) == 'eloftr':
        from gnss_denied_nav.localization.eloftr import EfficientLoFTRMatcher
        return EfficientLoFTRMatcher(scale=extractor.params.get('scale', 1.0), device=extractor.device)
    if getattr(extractor, 'matcher', None) == 'lightglue':
        from gnss_denied_nav.localization.learned import LightGlueMatcher
        return LightGlueMatcher('superpoint', device=extractor.device)
    return DescriptorMatcher(binary=extractor.binary, ratio=ratio)

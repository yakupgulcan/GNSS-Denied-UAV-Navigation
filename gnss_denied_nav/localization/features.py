"""Local feature extractors.

Every extractor exposes the same interface:

    extractor.name    -> method name stored in the map database ("SIFT", ...)
    extractor.binary  -> True for Hamming-distance descriptors
    extractor.params  -> dict of settings, stored in the database metadata
    extractor.extract(gray_image) -> Features

Learned extractors (e.g. SuperPoint) plug in by implementing the same
interface and registering in EXTRACTORS.
"""
import cv2
import numpy as np

from gnss_denied_nav.localization.types import Features


class OpenCVExtractor:
    """SIFT / ORB / AKAZE / BRISK via OpenCV, with one shared code path."""

    DEFAULTS = {
        'SIFT': {'n_features': 2000},
        'ORB': {'n_features': 2000},
        'AKAZE': {'threshold': 0.001},
        'BRISK': {'threshold': 60, 'octaves': 0},
    }

    def __init__(self, name, **params):
        name = name.upper()
        if name not in self.DEFAULTS:
            raise ValueError(f'Unknown OpenCV method {name!r}; '
                             f'supported: {sorted(self.DEFAULTS)}')
        self.name = name
        self.params = {**self.DEFAULTS[name], **params}
        self.binary = name != 'SIFT'
        self.clahe = None

        p = self.params
        if name == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=int(p['n_features']))
        elif name == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=int(p['n_features']))
        elif name == 'AKAZE':
            self.detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                threshold=float(p['threshold']),
                nOctaves=4,
                nOctaveLayers=4,
                diffusivity=cv2.KAZE_DIFF_PM_G2)
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        elif name == 'BRISK':
            self.detector = cv2.BRISK_create(thresh=int(p['threshold']),
                                             octaves=int(p['octaves']))

    def extract(self, gray):
        h, w = gray.shape[:2]
        img = self.clahe.apply(gray) if self.clahe is not None else gray
        kps, desc = self.detector.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            dim, dtype = (128, np.float32) if not self.binary else (32, np.uint8)
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, dim), dtype), (w, h))
        kpts = np.float32([k.pt for k in kps])
        return Features(kpts, desc, (w, h))


EXTRACTORS = {name: OpenCVExtractor for name in OpenCVExtractor.DEFAULTS}


def create_extractor(name, **params):
    name = name.upper()
    if name not in EXTRACTORS:
        raise ValueError(f'Unknown feature method {name!r}; supported: {sorted(EXTRACTORS)}')
    return EXTRACTORS[name](name, **params)

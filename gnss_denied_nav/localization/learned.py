"""Learned features and matchers (SuperPoint + LightGlue).

Imported lazily by ``features.create_extractor`` and ``matchers.create_matcher``
so that torch and lightglue are only needed when a learned method is used.

Model weights are downloaded by lightglue on first use into the torch hub
cache (~/.cache/torch/hub/checkpoints).
"""
from collections import OrderedDict

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from gnss_denied_nav.localization.types import Correspondences, Features

_EMPTY = Correspondences(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))


def resolve_device(device=None):
    if device in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class SuperPointExtractor:
    """SuperPoint keypoints and 256-D descriptors, matched with LightGlue.

    Descriptors are returned as float16 to halve the map database size; the
    matcher converts them back to float32.
    """

    name = 'SUPERPOINT'
    binary = False
    rotation_invariant = False  # Localizer turns the query by 90-deg steps using a heading prior
    matcher = 'lightglue'
    DEFAULTS = {'n_features': 1024, 'detection_threshold': 0.0005}

    def __init__(self, device=None, **params):
        # The device is a runtime choice, not a map property: keep it out of params.
        self.params = {**self.DEFAULTS, **params}
        self.device = resolve_device(device)
        self.model = SuperPoint(
            max_num_keypoints=int(self.params['n_features']),
            detection_threshold=float(self.params['detection_threshold']),
        ).eval().to(self.device)

    @torch.no_grad()
    def extract(self, gray):
        h, w = gray.shape[:2]
        img = torch.from_numpy(gray).to(self.device, torch.float32).div_(255.0)[None, None]
        feats = self.model.extract(img, resize=None)
        kpts = feats['keypoints'][0].cpu().numpy().astype(np.float32)
        desc = feats['descriptors'][0].cpu().numpy().astype(np.float16)
        return Features(kpts, desc, (w, h))


class LightGlueMatcher:
    """LightGlue matching of two Features sets (keypoints in pixels).

    Mixed precision is used on CUDA. The GPU tensors of the last query and of
    recently used map frames are cached, since a query is matched against
    several map frames and the same map frames recur from frame to frame.
    """

    def __init__(self, features='superpoint', device=None, cache_size=64, **conf):
        self.device = resolve_device(device)
        conf.setdefault('mp', self.device.type == 'cuda')
        self.model = LightGlue(features=features, **conf).eval().to(self.device)
        self.cache_size = cache_size
        self._cache = OrderedDict()   # map frames: key -> (base array, tensors)
        self._last = (None, None)     # last query: (Features, tensors)

    def _cached(self, f):
        base = f.kpts.base
        if base is None:
            # A standalone array (a query): reuse only for the very same object,
            # which the Localizer passes for every candidate of one frame.
            if self._last[0] is not f:
                self._last = (f, self._tensors(f))
            return self._last[1]
        # Map frame Features are views into the database arrays: identify a
        # frame by its base array and offset. The entry keeps the base alive,
        # so its id cannot be reused while cached.
        key = (id(base), f.kpts.__array_interface__['data'][0], len(f))
        entry = self._cache.get(key)
        if entry is None:
            entry = (base, self._tensors(f))
            self._cache[key] = entry
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return entry[1]

    def _tensors(self, f):
        return {
            'keypoints': torch.from_numpy(np.ascontiguousarray(f.kpts, np.float32))[None].to(self.device),
            'descriptors': torch.from_numpy(np.ascontiguousarray(f.desc, np.float32))[None].to(self.device),
            'image_size': torch.tensor([f.image_size], dtype=torch.float32, device=self.device),
        }

    @torch.no_grad()
    def match(self, query, ref):
        if len(query) < 2 or len(ref) < 2:
            return _EMPTY
        out = self.model({'image0': self._cached(query), 'image1': self._cached(ref)})
        pairs = out['matches'][0].cpu().numpy()
        if len(pairs) == 0:
            return _EMPTY
        return Correspondences(query.kpts[pairs[:, 0]], ref.kpts[pairs[:, 1]])

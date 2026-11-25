"""Efficient LoFTR (detector-free, semi-dense matching), official implementation.

Efficient LoFTR does not detect keypoints: both images go through the network
together and it returns matched pixel pairs on a coarse grid (8 px), refined
to sub-pixel. The "extractor" therefore only passes the image along, the map
database stores the survey images instead of features, and the matcher
returns ordinary Correspondences, so homography verification and the metric
pose work as for keypoint methods.

The official code (https://github.com/zju3dv/EfficientLoFTR) is used for
inference only. It is not a pip package: clone it and put the weights in
``<root>/weights/eloftr_outdoor.ckpt``; ``root`` is ``$ELOFTR_ROOT`` or
``~/third_party/EfficientLoFTR``. Only torch, einops, yacs, loguru and kornia
are needed (not the repository's training requirements).

The Hugging Face transformers port was not used: in transformers 5.2 its
outputs pair points wrongly under rotation and lose the sub-pixel refinement.

Imported lazily by ``features.create_extractor`` and ``matchers.create_matcher``.
"""
import os
import re
import sys
import types
from copy import deepcopy

import cv2
import numpy as np
import torch

from gnss_denied_nav.localization.types import Correspondences, Features

DEFAULT_ROOT = os.path.expanduser('~/third_party/EfficientLoFTR')
WEIGHTS = os.path.join('weights', 'eloftr_outdoor.ckpt')
_EMPTY = Correspondences(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))


def resolve_device(device=None):
    if device in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _import_official(root):
    """Import ``src.loftr`` from the official repository at ``root``."""
    if not os.path.isdir(os.path.join(root, 'src', 'loftr')):
        raise FileNotFoundError(
            f'Efficient LoFTR code not found in {root}. Clone '
            'https://github.com/zju3dv/EfficientLoFTR there or set ELOFTR_ROOT.')
    loaded = sys.modules.get('src')
    if loaded is not None and not os.path.abspath(getattr(loaded, '__file__', '') or '').startswith(
            os.path.abspath(root)):
        raise ImportError(f"Another 'src' package is already imported ({loaded}); "
                          'cannot import the Efficient LoFTR code.')
    if root not in sys.path:
        sys.path.insert(0, root)
    # src/utils/misc.py imports joblib and pytorch_lightning (training only);
    # the model needs just detect_NaN from it.
    if 'src.utils.misc' not in sys.modules:
        misc = types.ModuleType('src.utils.misc')

        def detect_NaN(feat_0, feat_1):  # noqa: N802 (name used by the official code)
            feat_0[torch.isnan(feat_0)] = 0
            feat_1[torch.isnan(feat_1)] = 0

        misc.detect_NaN = detect_NaN
        sys.modules['src.utils.misc'] = misc
    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
    return LoFTR, {'full': full_default_cfg, 'opt': opt_default_cfg}, reparameter


def _load_state_dict(path):
    """Read only the weights from the PyTorch Lightning checkpoint.

    The checkpoint also pickles Lightning objects. It is loaded with
    weights_only=True; the one Lightning class it references is replaced by an
    inert stand-in, so no code from the file runs and Lightning is not needed.
    """
    stand_ins = []
    for _ in range(10):
        try:
            with torch.serialization.safe_globals(stand_ins):
                ckpt = torch.load(path, map_location='cpu', weights_only=True)
            break
        except Exception as e:  # pickle.UnpicklingError with the offending global
            m = re.search(r'Unsupported global: GLOBAL ([\w.]+)', str(e))
            if m is None or not m.group(1).startswith('pytorch_lightning.'):
                raise
            stand_ins.append(_stand_in(m.group(1)))
    else:
        raise RuntimeError(f'Could not read {path}')
    return {k.removeprefix('matcher.'): v for k, v in ckpt['state_dict'].items()}


def _stand_in(qualname):
    module, name = qualname.rsplit('.', 1)
    parts = module.split('.')
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault('.'.join(parts[:i]), types.ModuleType('.'.join(parts[:i])))
    cls = type(name, (), {'__module__': module, '__setstate__': lambda self, state: None})
    setattr(sys.modules[module], name, cls)
    return cls


class ImageExtractor:
    """Passes the image to a detector-free matcher (no features are extracted).

    params['scale'] < 1 matches at a lower resolution (faster); points are
    scaled back, so the pose is still computed in full-resolution pixels.
    """

    name = 'ELOFTR'
    binary = False
    rotation_invariant = False
    image_based = True
    matcher = 'eloftr'
    DEFAULTS = {'scale': 1.0}

    def __init__(self, device=None, **params):
        self.params = {**self.DEFAULTS, **params}
        self.device = resolve_device(device)

    def extract(self, gray):
        h, w = gray.shape[:2]
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 0), np.float32), (w, h), gray)


class EfficientLoFTRMatcher:
    """Match two image-carrying Features with the official Efficient LoFTR.

    model_type: 'full' (default) or 'opt'. precision: 'fp16' (default on
    CUDA; measured: same accuracy as fp32, ~1.7x faster), 'mp' or 'fp32'.
    Images are scaled by ``scale`` and cropped to multiples of 32 pixels.
    """

    def __init__(self, scale=1.0, model_type='full', precision=None, device=None, root=None):
        self.device = resolve_device(device)
        self.scale = float(scale)
        if precision is None:
            precision = 'fp16' if self.device.type == 'cuda' else 'fp32'
        self.precision = precision
        root = os.path.expanduser(root or os.environ.get('ELOFTR_ROOT') or DEFAULT_ROOT)
        LoFTR, configs, reparameter = _import_official(root)
        cfg = deepcopy(configs[model_type])
        cfg['mp'] = precision == 'mp'
        cfg['half'] = precision == 'fp16'
        weights = os.path.join(root, WEIGHTS)
        if not os.path.exists(weights):
            raise FileNotFoundError(f'Efficient LoFTR weights not found: {weights}')
        model = LoFTR(config=cfg)
        model.load_state_dict(_load_state_dict(weights))
        model = reparameter(model)
        if precision == 'fp16':
            model = model.half()
        self.model = model.eval().to(self.device)
        self.dtype = torch.float16 if precision == 'fp16' else torch.float32

    def _tensor(self, img):
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        img = img[:h // 32 * 32, :w // 32 * 32]   # crop: pixel coordinates are unchanged
        t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device, torch.float32).div_(255.0)
        return t[None, None].to(self.dtype)

    @torch.no_grad()
    def match(self, query, ref):
        if query.image is None or ref.image is None:
            raise ValueError('Efficient LoFTR needs image-based Features '
                             '(build the map with --method ELOFTR)')
        batch = {'image0': self._tensor(query.image), 'image1': self._tensor(ref.image)}
        with torch.autocast(device_type=self.device.type, enabled=self.precision == 'mp'):
            self.model(batch)
        pts_q = batch['mkpts0_f'].float().cpu().numpy() / self.scale
        pts_r = batch['mkpts1_f'].float().cpu().numpy() / self.scale
        if len(pts_q) == 0:
            return _EMPTY
        return Correspondences(pts_q.astype(np.float32), pts_r.astype(np.float32))

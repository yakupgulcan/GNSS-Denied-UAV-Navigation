"""Georeferenced map database: survey frames with their poses and features.

File format (``.npz``, no pickled objects):

    meta        JSON string: format version, feature method + params, image size
    filenames   (M,) str
    lat, lon    (M,) float64   GPS tag of each frame
    heading     (M,) float64   compass heading (deg)
    rel_alt     (M,) float64   height above take-off (m)
    offsets     (M+1,) int64   frame i owns rows offsets[i]:offsets[i+1]
    kpts        (N, 2) float32 keypoints of all frames, concatenated
    desc        (N, D)         descriptors of all frames, concatenated

Image-based methods (Efficient LoFTR) store no features (N = 0); the meta
then has ``image_based: true`` and ``image_dir``, and features(i) returns the
survey image itself, read from image_dir (default: the database's directory).

Databases written by the old ``build_feature_database_*.py`` scripts
(pickled object arrays) are still readable; their method is inferred from
the descriptor type and rel_alt is taken from frames_details.json.
"""
import json
import os
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np

from gnss_denied_nav import geo
from gnss_denied_nav.localization.types import Features

FORMAT_VERSION = 1


class MapDatabase:

    def __init__(self, method, params, image_size, filenames, lat, lon, heading, rel_alt,
                 kpts, desc, offsets, notes=None, image_based=False, image_dir=None):
        self.method = method
        self.params = params
        self.image_size = tuple(image_size)
        self.filenames = np.asarray(filenames)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.heading = np.asarray(heading, dtype=np.float64)
        self.rel_alt = np.asarray(rel_alt, dtype=np.float64)
        self.kpts = kpts
        self.desc = desc
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.notes = notes or []
        self.enu = None
        self.image_based = bool(image_based)
        self.image_dir = image_dir
        self._images = OrderedDict()   # small LRU cache of survey images (image-based maps)

    def __len__(self):
        return len(self.filenames)

    # ------------------------------------------------------------------ access
    def features(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        image = self.image(i) if self.image_based else None
        return Features(self.kpts[a:b], self.desc[a:b], self.image_size, image)

    def image(self, i, cache_size=64):
        """Grayscale survey image of frame i (image-based maps), cached."""
        img = self._images.get(i)
        if img is None:
            path = os.path.join(self.image_dir or '', str(self.filenames[i]))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f'Survey image not found: {path} (set image_dir)')
            self._images[i] = img
            if len(self._images) > cache_size:
                self._images.popitem(last=False)
        else:
            self._images.move_to_end(i)
        return img

    def set_origin(self, origin_lat, origin_lon):
        """Compute each frame's local ENU position relative to the origin."""
        east, north = geo.latlon_to_enu(self.lat, self.lon, origin_lat, origin_lon)
        self.enu = np.column_stack([east, north])

    def frames_near(self, x, y, radius, max_count, exclude=()):
        """Indices of frames within radius of (x, y), nearest first."""
        d = np.hypot(self.enu[:, 0] - x, self.enu[:, 1] - y)
        idx = np.flatnonzero(d <= radius)
        if len(exclude):
            idx = idx[~np.isin(idx, exclude)]
        return idx[np.argsort(d[idx])][:max_count]

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, frames_dir, extractor, progress=None):
        """Extract features from every usable survey frame in frames_dir."""
        with open(os.path.join(frames_dir, 'frames_details.json')) as f:
            entries = json.load(f)

        image_based = getattr(extractor, 'image_based', False)
        names, lat, lon, heading, alt, kpts, desc = [], [], [], [], [], [], []
        image_size = None
        for n, e in enumerate(entries):
            gps = e.get('gps') or {}
            if gps.get('lat') is None or e.get('heading') is None or e.get('rel_alt') is None:
                continue
            img = cv2.imread(os.path.join(frames_dir, e['filename']), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feats = extractor.extract(img)
            if len(feats) < 10 and not image_based:
                continue
            if image_size is None:
                image_size = feats.image_size
            elif feats.image_size != image_size:
                raise ValueError(f"{e['filename']}: image size {feats.image_size} "
                                 f"differs from {image_size}")
            names.append(e['filename'])
            lat.append(gps['lat'])
            lon.append(gps['lon'])
            heading.append(e['heading'])
            alt.append(e['rel_alt'])
            kpts.append(feats.kpts)
            desc.append(feats.desc)
            if progress:
                progress(n + 1, len(entries))

        if not names:
            raise ValueError(f'No usable frames in {frames_dir}')
        offsets = np.concatenate([[0], np.cumsum([len(k) for k in kpts])])
        return cls(extractor.name, extractor.params, image_size, names, lat, lon, heading, alt,
                   np.vstack(kpts).astype(np.float32), np.vstack(desc), offsets,
                   image_based=image_based, image_dir=os.path.abspath(frames_dir))

    def save(self, path):
        meta = {
            'version': FORMAT_VERSION,
            'method': self.method,
            'params': self.params,
            'image_size': list(self.image_size),
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        if self.image_based:
            meta.update(image_based=True, image_dir=self.image_dir)
        np.savez_compressed(
            path, meta=np.array(json.dumps(meta)), filenames=self.filenames.astype(str),
            lat=self.lat, lon=self.lon, heading=self.heading, rel_alt=self.rel_alt,
            offsets=self.offsets, kpts=self.kpts, desc=self.desc)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path):
        path = os.path.expanduser(path)
        with np.load(path, allow_pickle=False) as z:
            if 'meta' in z.files:
                meta = json.loads(str(z['meta']))
                if meta.get('version') != FORMAT_VERSION:
                    raise ValueError(f"{path}: unsupported database version {meta.get('version')}")
                image_dir = meta.get('image_dir')
                if meta.get('image_based') and not (image_dir and os.path.isdir(image_dir)):
                    image_dir = os.path.dirname(os.path.abspath(path))  # moved with its frames
                return cls(meta['method'], meta['params'], meta['image_size'],
                           z['filenames'], z['lat'], z['lon'], z['heading'], z['rel_alt'],
                           z['kpts'], z['desc'], z['offsets'],
                           image_based=meta.get('image_based', False), image_dir=image_dir)
        return cls._load_legacy(path)

    @classmethod
    def _load_legacy(cls, path, image_size=(640, 480), default_alt=30.0):
        notes = [f'{os.path.basename(path)} is a legacy database; rebuild it with build_map_db.']
        with np.load(path, allow_pickle=True) as z:
            if 'descriptors' not in z.files:
                raise ValueError(f'{path}: not a keypoint database (HOG databases are not supported)')
            descs, kps, names, gps = z['descriptors'], z['keypoints'], z['filenames'], z['gps_data']
            headings = z['headings'] if 'headings' in z.files else np.zeros(len(names))

        alt_by_name = {}
        details = os.path.join(os.path.dirname(path), 'frames_details.json')
        if os.path.exists(details):
            with open(details) as f:
                alt_by_name = {e['filename']: e.get('rel_alt') for e in json.load(f)}
        else:
            notes.append(f'frames_details.json not found; assuming rel_alt={default_alt} m')

        keep = [i for i in range(len(names))
                if descs[i] is not None and len(descs[i]) >= 10
                and isinstance(gps[i], dict) and gps[i].get('lat') is not None]
        sample = descs[keep[0]]
        if sample.dtype == np.float32:
            method = 'SIFT'
        else:
            method = {32: 'ORB', 64: 'BRISK', 61: 'AKAZE'}.get(sample.shape[1])
            if method is None:
                raise ValueError(f'{path}: cannot infer method from descriptor width {sample.shape[1]}')

        kpts = [np.asarray(kps[i], np.float32)[:, :2] for i in keep]
        offsets = np.concatenate([[0], np.cumsum([len(k) for k in kpts])])
        return cls(method, {}, image_size,
                   [str(names[i]) for i in keep],
                   [gps[i]['lat'] for i in keep], [gps[i]['lon'] for i in keep],
                   [float(headings[i]) for i in keep],
                   [alt_by_name.get(str(names[i])) or default_alt for i in keep],
                   np.vstack(kpts), np.vstack([descs[i] for i in keep]), offsets, notes)

"""Localization core tests on a synthetic nadir world (no dataset or ROS needed)."""
import math

import cv2
import numpy as np
import pytest

from gnss_denied_nav import geo
from gnss_denied_nav.localization import (
    CameraModel, Localizer, LocalizerConfig, MapDatabase, create_extractor)

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


def render(ground, east, north, heading_deg):
    """Nadir camera image at ENU (east, north), image-up = heading."""
    gsd = CAM.gsd(ALT)
    s = gsd / GROUND_RES                                   # ground px per image px
    c = ground.shape[0] / 2.0
    gx, gy = c + east / GROUND_RES, c - north / GROUND_RES  # ground image: x east, y south
    h = math.radians(heading_deg)
    # image (u right, v down) -> ground px; image-up points along the heading
    fwd = np.array([math.sin(h), -math.cos(h)])            # ground-px direction of image-up
    right = np.array([math.cos(h), math.sin(h)])
    cx, cy = CAM.center
    M = np.column_stack([right * s, -fwd * s])
    t = np.array([gx, gy]) - M @ np.array([cx, cy])
    A = np.hstack([M, t[:, None]])
    return cv2.warpAffine(ground, A, (CAM.width, CAM.height),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def build_db(ground, poses):
    ext = create_extractor('SIFT', n_features=1500)
    feats = [ext.extract(render(ground, *p)) for p in poses]
    lat, lon = zip(*[geo.enu_to_latlon(e, n, *ORIGIN) for e, n, _ in poses])
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in feats])])
    db = MapDatabase('SIFT', ext.params, (CAM.width, CAM.height),
                     [f'f{i}.jpg' for i in range(len(poses))], lat, lon,
                     [p[2] for p in poses], [ALT] * len(poses),
                     np.vstack([f.kpts for f in feats]), np.vstack([f.desc for f in feats]), offsets)
    db.set_origin(*ORIGIN)
    return db


@pytest.fixture(scope='module')
def world():
    ground = make_ground()
    db = build_db(ground, [(0.0, 0.0, 0.0), (0.0, 30.0, 0.0), (20.0, 15.0, 90.0)])
    return ground, db


@pytest.mark.parametrize('east,north,heading', [
    (3.0, 8.0, 0.0),
    (-4.0, 22.0, 15.0),
    (15.0, 12.0, 100.0),
])
def test_prior_localization_is_metric(world, east, north, heading):
    ground, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, info = loc.localize(render(ground, east, north, heading), prior=(0.0, 10.0), radius=50.0)
    assert res is not None, info
    assert math.hypot(res.x - east, res.y - north) < 0.5
    assert abs((res.heading_deg - heading + 180) % 360 - 180) < 1.0
    assert res.scale == pytest.approx(1.0, abs=0.05)


def test_global_retrieval_without_prior(world):
    ground, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=True))
    res, info = loc.localize(render(ground, 1.0, 27.0, 5.0))
    assert res is not None, info
    assert math.hypot(res.x - 1.0, res.y - 27.0) < 0.5


def test_unrelated_image_is_rejected(world):
    _, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    other = render(make_ground(seed=1), 0.0, 0.0, 0.0)
    res, _ = loc.localize(other, prior=(0.0, 0.0), radius=100.0)
    assert res is None


def test_database_roundtrip(world, tmp_path):
    _, db = world
    path = tmp_path / 'map.npz'
    db.save(path)
    db2 = MapDatabase.load(str(path))
    assert db2.method == 'SIFT' and len(db2) == len(db)
    np.testing.assert_array_equal(db2.offsets, db.offsets)
    np.testing.assert_allclose(db2.lat, db.lat)
    np.testing.assert_array_equal(db2.features(1).desc, db.features(1).desc)

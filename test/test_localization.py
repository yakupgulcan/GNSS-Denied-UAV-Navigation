"""Localization core tests on a synthetic nadir world (no dataset or ROS needed)."""
import math

import numpy as np
import pytest
from synthetic import ALT, CAM, ORIGIN, make_ground, render

from gnss_denied_nav import geo
from gnss_denied_nav.localization import (
    Localizer, LocalizerConfig, MapDatabase, create_extractor)


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


@pytest.mark.parametrize('k', [1, 2, 3])
def test_unrotate_points_inverts_cv2_rotate(k):
    import cv2
    from gnss_denied_nav.localization.localizer import _ROTATE, unrotate_points
    h, w = 48, 64
    img = np.zeros((h, w), np.uint8)
    pts = np.float32([[5, 7], [60, 40], [33, 2]])
    for x, y in pts.astype(int):
        img[y, x] = 255
    turned = cv2.rotate(img, _ROTATE[k])
    ys, xs = np.nonzero(turned)
    back = unrotate_points(np.float32(np.column_stack([xs, ys])), k, w, h)
    assert sorted(map(tuple, back.astype(int))) == sorted(map(tuple, pts.astype(int)))


def test_rotation_steps():
    from gnss_denied_nav.localization.localizer import rotation_steps
    assert rotation_steps(0.0, 0.0) == 0
    assert rotation_steps(92.0, 0.0) == 1
    assert rotation_steps(178.0, 359.0) == 2
    assert rotation_steps(0.0, 85.0) == 3


def test_heading_gate_rejects_inconsistent_fix(world):
    ground, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    img = render(ground, 3.0, 8.0, 0.0)
    res, _ = loc.localize(img, prior=(0.0, 10.0), radius=50.0, heading=5.0)
    assert res is not None
    res, info = loc.localize(img, prior=(0.0, 10.0), radius=50.0, heading=90.0)
    assert res is None and info['rejects'].get('heading', 0) > 0


def test_accept_inliers_stops_at_first_strong_fix(world):
    ground, db = world
    img = render(ground, 3.0, 8.0, 0.0)
    all_ = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res_all, info_all = all_.localize(img, prior=(0.0, 0.0), radius=50.0)
    early = Localizer(db, CAM, LocalizerConfig(global_retrieval=False, accept_inliers=20))
    res, info = early.localize(img, prior=(0.0, 0.0), radius=50.0)
    assert info_all['n_tried'] == 3 and info['n_tried'] == 1
    assert res is not None and math.hypot(res.x - 3.0, res.y - 8.0) < 0.5

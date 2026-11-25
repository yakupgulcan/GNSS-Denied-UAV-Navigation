"""Efficient LoFTR behind the localization interfaces (synthetic world).

Skipped unless the official code and weights are installed (see README,
"Learned matchers").
"""
import math
import os

import numpy as np
import pytest
from synthetic import ALT, CAM, ORIGIN, make_ground, render

pytest.importorskip('einops')
pytest.importorskip('yacs')
_ROOT = os.path.expanduser(os.environ.get('ELOFTR_ROOT', '~/third_party/EfficientLoFTR'))
if not os.path.exists(os.path.join(_ROOT, 'weights', 'eloftr_outdoor.ckpt')):
    pytest.skip(f'Efficient LoFTR code/weights not found in {_ROOT}', allow_module_level=True)

from gnss_denied_nav import geo  # noqa: E402
from gnss_denied_nav.localization import (  # noqa: E402
    Localizer, LocalizerConfig, MapDatabase, create_extractor)

POSES = [(0.0, 0.0, 0.0), (0.0, 30.0, 0.0), (20.0, 15.0, 90.0)]


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    ground = make_ground()
    frames = tmp_path_factory.mktemp('frames')
    import cv2
    names = []
    for i, p in enumerate(POSES):
        names.append(f'f{i}.jpg')
        cv2.imwrite(str(frames / names[-1]), render(ground, *p))
    ext = create_extractor('ELOFTR')
    lat, lon = zip(*[geo.enu_to_latlon(e, n, *ORIGIN) for e, n, _ in POSES])
    db = MapDatabase(ext.name, ext.params, (CAM.width, CAM.height), names, lat, lon,
                     [p[2] for p in POSES], [ALT] * len(POSES), np.zeros((0, 2), np.float32),
                     np.zeros((0, 0), np.float32), np.zeros(len(POSES) + 1, np.int64),
                     image_based=True, image_dir=str(frames))
    db.set_origin(*ORIGIN)
    return ground, db, frames


def test_database_round_trip_keeps_images(world, tmp_path):
    _, db, frames = world
    path = tmp_path / 'map.npz'
    db.save(path)
    loaded = MapDatabase.load(path)
    assert loaded.image_based and loaded.method == 'ELOFTR'
    assert loaded.features(0).image.shape == (CAM.height, CAM.width)


@pytest.mark.parametrize('east,north,heading', [
    (3.0, 8.0, 0.0),
    (2.0, 25.0, 180.0),   # turned towards the map frame by the heading prior
    (1.0, 27.0, 90.0),    # turned query is portrait, the map frame landscape
])
def test_prior_localization_is_metric(world, east, north, heading):
    ground, db, _ = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, info = loc.localize(render(ground, east, north, heading), prior=(east, north), radius=20.0,
                             heading=heading + 3.0)
    assert res is not None, info
    assert math.hypot(res.x - east, res.y - north) < 0.5
    assert abs((res.heading_deg - heading + 180) % 360 - 180) < 1.0


def test_unrelated_image_is_rejected(world):
    _, db, _ = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, _ = loc.localize(render(make_ground(seed=1), 0.0, 0.0, 0.0), prior=(0.0, 0.0), radius=100.0)
    assert res is None

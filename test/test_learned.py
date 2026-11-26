"""SuperPoint + LightGlue behind the localization interfaces (synthetic world).

Skipped when lightglue is not installed. The first run downloads the model
weights into the torch hub cache.
"""
import math

import numpy as np
import pytest
from synthetic import ALT, CAM, ORIGIN, make_ground, render

pytest.importorskip('lightglue')

from gnss_denied_nav import geo  # noqa: E402
from gnss_denied_nav.localization import (  # noqa: E402
    Localizer, LocalizerConfig, MapDatabase, create_extractor)
from gnss_denied_nav.localization.learned import LightGlueMatcher  # noqa: E402
from gnss_denied_nav.localization.matchers import create_matcher  # noqa: E402


@pytest.fixture(scope='module')
def extractor():
    return create_extractor('SUPERPOINT', n_features=1024)


@pytest.fixture(scope='module')
def world(extractor):
    ground = make_ground()
    poses = [(0.0, 0.0, 0.0), (0.0, 30.0, 0.0), (20.0, 15.0, 90.0)]
    feats = [extractor.extract(render(ground, *p)) for p in poses]
    lat, lon = zip(*[geo.enu_to_latlon(e, n, *ORIGIN) for e, n, _ in poses])
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in feats])])
    db = MapDatabase(extractor.name, extractor.params, (CAM.width, CAM.height),
                     [f'f{i}.jpg' for i in range(len(poses))], lat, lon,
                     [p[2] for p in poses], [ALT] * len(poses),
                     np.vstack([f.kpts for f in feats]), np.vstack([f.desc for f in feats]), offsets)
    db.set_origin(*ORIGIN)
    return ground, db


def test_superpoint_features(extractor, world):
    ground, _ = world
    f = extractor.extract(render(ground, 0.0, 0.0, 0.0))
    assert 100 < len(f) <= 1024
    assert f.desc.shape == (len(f), 256) and f.desc.dtype == np.float16
    assert f.image_size == (CAM.width, CAM.height)
    assert 'device' not in extractor.params  # runtime choice, not stored in the map


def test_create_matcher_picks_lightglue(extractor):
    assert isinstance(create_matcher(extractor), LightGlueMatcher)


def test_database_round_trip(tmp_path, world):
    _, db = world
    path = tmp_path / 'map.npz'
    db.save(path)
    loaded = MapDatabase.load(path)
    assert loaded.method == 'SUPERPOINT'
    assert loaded.desc.dtype == np.float16
    assert loaded.params == db.params


@pytest.mark.parametrize('east,north,heading', [
    (3.0, 8.0, 0.0),
    (15.0, 12.0, 100.0),
])
def test_prior_localization_is_metric(world, east, north, heading):
    ground, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, info = loc.localize(render(ground, east, north, heading), prior=(0.0, 10.0), radius=50.0)
    assert res is not None, info
    assert math.hypot(res.x - east, res.y - north) < 0.5
    assert abs((res.heading_deg - heading + 180) % 360 - 180) < 1.0


def test_unrelated_image_is_rejected(world):
    _, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, _ = loc.localize(render(make_ground(seed=1), 0.0, 0.0, 0.0), prior=(0.0, 0.0), radius=100.0)
    assert res is None


@pytest.mark.parametrize('heading', [90.0, 180.0, 270.0])
def test_query_is_turned_towards_map_heading(world, heading):
    # The map frames near the query are at heading 0; SuperPoint is not rotation
    # invariant, so the heading prior is what makes these fixes possible.
    ground, db = world
    loc = Localizer(db, CAM, LocalizerConfig(global_retrieval=False))
    res, info = loc.localize(render(ground, 2.0, 25.0, heading), prior=(0.0, 25.0), radius=20.0,
                             heading=heading + 4.0)
    assert res is not None, info
    assert math.hypot(res.x - 2.0, res.y - 25.0) < 0.5
    assert abs((res.heading_deg - heading + 180) % 360 - 180) < 1.0

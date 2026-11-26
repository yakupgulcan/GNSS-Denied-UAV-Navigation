"""Run naming and metadata (no simulator needed)."""
import json

from gnss_denied_nav.localization.database import MapDatabase, method_tags
from gnss_denied_nav.run_logging import describe_run, launch_overrides, run_name


def test_method_tags():
    assert method_tags('SIFT', {'n_features': 2000}) == ['sift', 'n2000']
    assert method_tags('ELOFTR', {'scale': 1.0}) == ['eloftr', 's1']
    assert method_tags('BRISK', {'threshold': 60, 'octaves': 0}) == ['brisk', 't60', 'o0']


def test_launch_overrides_only_given_arguments():
    out = launch_overrides({'db_path': ' ', 'target_y': '1000', 'target_x': ''})
    assert out['visual_estimator_node'] == {}
    assert out['guided_mission'] == {'target_pos_y': 1000.0}


def _write_db(path):
    import numpy as np
    meta = {'version': 1, 'method': 'ORB', 'params': {'n_features': 2000}, 'image_size': [640, 480]}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)))


def test_describe_run_and_name(tmp_path):
    db = tmp_path / 'map_orb_n2000.npz'
    _write_db(db)
    params = tmp_path / 'params.yaml'
    params.write_text('visual_estimator_node:\n  ros__parameters:\n    db_path: "other.npz"\n'
                      'guided_mission:\n  ros__parameters:\n    target_pos_x: 0.0\n'
                      '    target_pos_y: 2900.0\n    target_altitude: 30.0\n')
    cfg = {'params_file': str(params), 'world': 'map_changed.sdf', 'db_path': str(db),
           'target_y': '1000', 'tag': 'r 1', 'note': 'first try'}
    meta = describe_run(cfg, 'guided', 'guided_mission')

    assert MapDatabase.read_meta(str(db))['method'] == 'ORB'
    assert meta['map']['label'] == 'orb-n2000' and len(meta['map']['db_sha256']) == 64
    assert meta['route']['distance_m'] == 1000.0
    assert meta['parameters']['visual_estimator_node']['db_path'] == str(db)
    assert meta['note'] == 'first try'
    assert run_name(meta, 'STAMP') == 'STAMP_map_changed_orb-n2000_1km_r-1'
    json.dumps(meta)  # serializable


def test_missing_database_is_reported(tmp_path):
    meta = describe_run({'world': 'w.sdf', 'db_path': str(tmp_path / 'none.npz')}, 'guided', 'guided_mission')
    assert meta['map']['method'] == 'unknown' and 'error' in meta['map']
    assert run_name(meta, 'S') == 'S_w_unknown_route'


def test_localizer_overrides_in_name(tmp_path):
    db = tmp_path / 'map_orb_n2000.npz'
    _write_db(db)
    cfg = {'world': 'map_changed.sdf', 'db_path': str(db), 'max_candidates': '5', 'accept_inliers': '0'}
    meta = describe_run(cfg, 'guided', 'guided_mission')
    assert meta['parameters']['visual_estimator_node']['max_candidates'] == 5
    assert meta['localizer_overrides'] == {'max_candidates': 5, 'accept_inliers': 0}
    assert run_name(meta, 'S') == 'S_map_changed_orb-n2000_route_c5-ai0'


def test_offline_settings_label():
    from types import SimpleNamespace as NS
    from gnss_denied_nav.tools.eval_localization import settings_label
    base = dict(candidates=3, accept_inliers=150, mode='prior', no_heading=False, heading_noise=5.0)
    assert settings_label(NS(**base)) == 'c3-ai150'
    assert settings_label(NS(**{**base, 'accept_inliers': 0, 'no_heading': True})) == 'c3-ai0-nohdg'
    assert settings_label(NS(**{**base, 'mode': 'global', 'heading_noise': 30.0})) == 'c3-ai150-global-hn30'

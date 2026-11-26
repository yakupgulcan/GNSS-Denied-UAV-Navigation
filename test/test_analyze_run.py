"""analyze_run on a synthetic flight with known errors (no bag or simulator needed)."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gnss_denied_nav import geo
from gnss_denied_nav.tools import analyze_run

LAT0, LON0 = -35.3766471, 149.1652375
T0 = 1000.0           # wall = sim time (RTF 1)
TAKEOFF, LAND, END = 10.0, 200.0, 220.0
SPEED, HEIGHT = 3.0, 30.0


def truth(t):
    """On the ground until TAKEOFF + 5 s, then north at SPEED and HEIGHT."""
    y = np.clip(t - TAKEOFF - 5.0, 0.0, None) * SPEED
    z = np.where(t > TAKEOFF + 2.0, HEIGHT, 0.0)
    return np.zeros_like(y), y, z


def stamp_of(t):
    return NS(stamp=NS(sec=int(t), nanosec=int((t % 1) * 1e9)))


def fix_msg(x, y):
    lat, lon = geo.enu_to_latlon(x, y, LAT0, LON0)
    return NS(latitude=lat, longitude=lon, status=NS(status=0))


def synthetic_bag():
    t_odo = np.arange(0.0, END, 0.02)
    x, y, z = truth(t_odo)
    odo = [(T0 + t, NS(header=stamp_of(T0 + t), pose=NS(pose=NS(position=NS(x=a, y=b, z=c)))))
           for t, a, b, c in zip(t_odo, x, y, z)]
    state = [(T0 + t, NS(armed=t >= TAKEOFF, mode='LAND' if t >= LAND else 'GUIDED'))
             for t in np.arange(0.0, END, 1.0)]
    t_pos = np.arange(5.0, END, 0.25)
    tx, ty, _ = truth(t_pos)
    ekf = [(T0 + t, fix_msg(a + 1.0, b)) for t, a, b in zip(t_pos, tx, ty)]        # 1 m east
    est = [(T0 + t, fix_msg(a, b - 2.0)) for t, a, b in zip(t_pos, tx, ty)]        # 2 m behind
    t_d = np.arange(TAKEOFF + 3.0, END, 0.1)
    dx, dy, _ = truth(t_d)
    diag = [(T0 + t, NS(data=json.dumps({'stamp': T0 + t, 'status': 'MATCH', 'ms_total': 40.0,
                                         'inliers': 300, 'n_candidates': 3, 'ref_index': 1,
                                         'fix_x': a, 'fix_y': b + 1.5})))
            for t, a, b in zip(t_d, dx, dy)]                                         # 1.5 m ahead
    gps = [(T0 + t, NS()) for t in np.arange(5.0, END, 0.2)]
    return {'/odometry': odo, '/mavros/state': state, '/mavros/global_position/global': ekf,
            '/visual_gps': est, '/visual_nav/diagnostics': diag,
            '/mavros/local_position/velocity_local': [], '/mavros/gps_input/gps_input': gps}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_run, 'read_bag', lambda _: synthetic_bag())
    meta = {'run': 'synthetic', 'world': 'flat', 'map': {'method': 'SIFT', 'label': 'sift-n2000'},
            'route': {'target_x': 0.0, 'target_y': 600.0, 'distance_m': 600.0},
            'parameters': {'guided_mission': {'start_lat': LAT0, 'start_lon': LON0}}}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    return tmp_path


def test_known_errors(run_dir):
    s = analyze_run.analyze(str(run_dir), plots=False)

    assert s['window']['end'] == 'LAND'
    assert s['window']['takeoff_s'] == pytest.approx(TAKEOFF, abs=1.0)
    assert s['window']['duration_s'] == pytest.approx(LAND - TAKEOFF, abs=1.0)

    # EKF 1 m east of a north-bound vehicle: cross-track -1 (right), along 0.
    assert s['ekf']['error']['median'] == pytest.approx(1.0, abs=0.02)
    assert s['ekf']['cross']['mean'] == pytest.approx(-1.0, abs=0.02)
    assert s['ekf']['along']['mean'] == pytest.approx(0.0, abs=0.05)
    # Estimate 2 m behind, raw fixes 1.5 m ahead.
    assert s['estimate']['along']['mean'] == pytest.approx(-2.0, abs=0.05)
    assert s['localization']['raw_fix']['along']['mean'] == pytest.approx(1.5, abs=0.05)

    rf = s['route_following']
    assert rf['truth_cross_track']['max_abs'] == pytest.approx(0.0, abs=1e-6)
    assert rf['ekf_cross_track']['mean'] == pytest.approx(-1.0, abs=0.02)
    # At the LAND switch the vehicle is (LAND - TAKEOFF - 5) * SPEED north.
    progress = (LAND - TAKEOFF - 5.0) * SPEED
    assert rf['progress_at_end_m'] == pytest.approx(progress, abs=1.0)
    assert rf['truth_to_target_at_end_m'] == pytest.approx(600.0 - progress, abs=1.0)

    assert s['localization']['accepted_ratio'] == 1.0
    assert s['gps_input']['sent_rate_hz'] == pytest.approx(5.0, abs=0.1)


def test_after_land_is_not_evaluated(run_dir):
    analyze_run.analyze(str(run_dir), plots=False)
    data = np.genfromtxt(run_dir / 'timeseries.csv', delimiter=',', names=True)
    after = data['t'] >= LAND + 0.5
    assert np.all(data['phase'][after] == 2)
    assert np.all(data['phase'][(data['t'] > TAKEOFF + 0.5) & (data['t'] < LAND - 0.5)] == 1)
    assert (run_dir / 'summary.json').exists() and (run_dir / 'localization.csv').exists()

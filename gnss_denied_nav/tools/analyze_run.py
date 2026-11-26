#!/usr/bin/env python3
"""Evaluate flights from their run directories, all with the same definitions.

Usage:
    ros2 run gnss_denied_nav analyze_run                      # newest run in ~/gnss_nav_logs
    ros2 run gnss_denied_nav analyze_run <run_dir> [<run_dir> ...]
    ros2 run gnss_denied_nav analyze_run --all                # every run with a bag
    ros2 run gnss_denied_nav analyze_run <run_dir> --no-plots

Reads the run's bag (and meta.json, mav.tlog when present) and writes into the
run directory:

    summary.json        all metrics (keys listed in the README)
    timeseries.csv      0.1 s grid: truth, EKF, published estimate, errors, phase
    localization.csv    one row per processed camera frame, with its error
    plots/*.png         trajectory, errors, route following, CDF, localization,
                        speed and altitude

Definitions:
- Evaluation window: from arming (take-off) to the switch to LAND. LAND and
  everything after it are not evaluated. Without a LAND switch the window
  ends at disarming or at the end of the recording (summary: window.end).
- Positions are ENU metres from home. Truth is Gazebo /odometry relative to
  its first pose. EKF is /mavros/global_position/global. The published
  estimate is /visual_gps, i.e. what goes to GPS_INPUT. Raw fix is each map
  fix at its image time (/visual_nav/diagnostics).
- Times are simulation time. Topics stamped with wall time are mapped through
  the /odometry receive times.
- Error is the distance to the truth at the same time. Along/cross are its
  components relative to the true direction of travel, during cruise only
  (speed > 1.5 m/s and height > 20 m). Positive along means ahead of the
  vehicle; positive cross means to the left.
- Route is the straight line from home to the target. Cross-track is the
  signed distance from it (positive = left of the route).
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

from gnss_denied_nav import geo

LOG_DIR = '~/gnss_nav_logs'
LAT0, LON0 = -35.3766471, 149.1652375
DT = 0.1                  # grid step (s)
CRUISE_SPEED = 1.5        # m/s
CRUISE_HEIGHT = 20.0      # m
MAX_GAP = 1.0             # s: grid values further than this from a sample are NaN
GPS_UNHEALTHY_DT = 0.245  # s: ArduPilot flags the GPS when updates are slower on average

TOPICS = ['/odometry', '/mavros/state', '/mavros/global_position/global',
          '/mavros/local_position/velocity_local', '/visual_gps', '/visual_nav/diagnostics',
          '/mavros/gps_input/gps_input']


# ------------------------------------------------------------------ input
def read_bag(bag_dir):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in TOPICS if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    classes = {t: get_message(types[t]) for t in present}
    msgs = {t: [] for t in TOPICS}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msgs[topic].append((t_ns * 1e-9, deserialize_message(data, classes[topic])))
    return msgs


def load_meta(run_dir):
    """meta.json, or one rebuilt from run_meta.json for runs recorded before it existed."""
    name = os.path.basename(os.path.normpath(run_dir))
    path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    old = os.path.join(run_dir, 'run_meta.json')
    if not os.path.exists(old):
        return {'run': name, 'world': 'unknown', 'map': {'method': 'unknown'}, 'route': {},
                'parameters': {}, 'reconstructed': True}
    from gnss_denied_nav.run_logging import describe_run
    with open(old) as f:
        r = json.load(f)
    cfg = dict(r.get('launch_arguments', {}))
    copied = os.path.join(run_dir, 'visual_nav_params.yaml')
    if os.path.exists(copied):
        cfg['params_file'] = copied
    meta = describe_run(cfg, 'guided', 'guided_mission')
    meta.update(run=name, started=r.get('started'), reconstructed=True,
                code={'note': 'recorded before meta.json; code state unknown'})
    return meta


def read_tlog(path):
    """GPS_INPUT arrival times at MAVProxy and the ArduPilot version (wall time)."""
    if not os.path.exists(path):
        return None
    try:
        from pymavlink import mavutil
    except ImportError:
        return None
    m = mavutil.mavlink_connection(path)
    t_gps, version = [], None
    while True:
        msg = m.recv_match(type=['GPS_INPUT', 'STATUSTEXT'])
        if msg is None:
            break
        if msg.get_type() == 'GPS_INPUT':
            t_gps.append(msg._timestamp)
        elif version is None and msg.text.startswith('ArduCopter V'):
            version = msg.text.strip()
    return {'gps_input_t': np.array(t_gps), 'version': version}


# ------------------------------------------------------------------ helpers
def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def num(v):
    """JSON-friendly float (None for NaN)."""
    v = float(v)
    return None if math.isnan(v) else round(v, 4)


def error_stats(e):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    if len(e) == 0:
        return {'n': 0}
    return {'n': int(len(e)), 'rmse': num(np.sqrt(np.mean(e ** 2))), 'mean': num(e.mean()),
            'median': num(np.median(e)), 'p90': num(np.percentile(e, 90)), 'max': num(e.max())}


def value_stats(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'mean': num(v.mean()), 'median': num(np.median(v)),
            'p90': num(np.percentile(v, 90)), 'max': num(v.max())}


def signed_stats(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'mean': num(v.mean()), 'std': num(v.std()),
            'rmse': num(np.sqrt(np.mean(v ** 2))), 'max_abs': num(np.abs(v).max())}


def on_grid(grid, t, v):
    """Interpolate samples (t, v) onto grid; NaN where no sample is within MAX_GAP."""
    out = np.full(len(grid), np.nan)
    if len(t) < 2:
        return out
    order = np.argsort(t)
    t, v = t[order], v[order]
    out[:] = np.interp(grid, t, v)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.minimum(np.abs(grid - t[idx - 1]), np.abs(t[idx] - grid))
    out[(near > MAX_GAP) | (grid < t[0]) | (grid > t[-1])] = np.nan
    return out


# ------------------------------------------------------------------ analysis
class Flight:
    """Truth, time mapping and evaluation window of one run."""

    def __init__(self, msgs):
        od = msgs['/odometry']
        if len(od) < 100:
            raise RuntimeError('not enough /odometry in the bag')
        self.tw = np.array([w for w, _ in od])
        self.ts = np.array([stamp(m) for _, m in od])
        p = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                      for _, m in od])
        p -= p[0]                                  # home = first pose, on the ground
        self.grid = np.arange(self.ts[0], self.ts[-1], DT)
        self.x, self.y, self.z = (np.interp(self.grid, self.ts, p[:, k]) for k in range(3))
        k = 5                                      # velocity smoothed over 0.5 s
        self.vx = np.convolve(np.gradient(self.x, DT), np.ones(k) / k, mode='same')
        self.vy = np.convolve(np.gradient(self.y, DT), np.ones(k) / k, mode='same')
        self.speed = np.hypot(self.vx, self.vy)

    def to_sim(self, wall):
        return np.interp(wall, self.tw, self.ts)

    def truth(self, t):
        return np.interp(t, self.grid, self.x), np.interp(t, self.grid, self.y)

    def along_cross(self, t, x, y):
        """Error components relative to the true direction of travel at t."""
        tx, ty = self.truth(t)
        ux, uy = np.interp(t, self.grid, self.vx), np.interp(t, self.grid, self.vy)
        n = np.maximum(np.hypot(ux, uy), 1e-6)
        ux, uy = ux / n, uy / n
        ex, ey = x - tx, y - ty
        return ex * ux + ey * uy, -ex * uy + ey * ux

    def set_window(self, msgs):
        """Take-off (arming) to the LAND switch, from /mavros/state."""
        st = msgs['/mavros/state']
        t = self.to_sim(np.array([w for w, _ in st]))
        armed = np.array([m.armed for _, m in st], bool)
        mode = [m.mode for _, m in st]
        if not armed.any():
            raise RuntimeError('the vehicle never armed')
        i0 = int(np.argmax(armed))
        self.t_takeoff = float(t[i0])
        self.end_reason, self.t_end = 'end of recording', float(self.ts[-1])
        for i in range(i0, len(t)):
            if mode[i] == 'LAND':
                self.end_reason, self.t_end = 'LAND', float(t[i])
                break
            if not armed[i]:
                self.end_reason, self.t_end = 'disarmed', float(t[i])
                break
        self.window = (self.grid >= self.t_takeoff) & (self.grid < self.t_end)
        self.cruise = self.window & (self.speed > CRUISE_SPEED) & (self.z > CRUISE_HEIGHT)

    def in_window(self, t):
        return (t >= self.t_takeoff) & (t < self.t_end)

    def in_cruise(self, t):
        return np.interp(t, self.grid, self.cruise.astype(float)) > 0.5


def position_series(f, msgs, topic, lat0, lon0):
    """(sim time, east, north) of a NavSatFix topic, without no-fix messages."""
    rows = [(w, m) for w, m in msgs[topic]
            if not math.isnan(m.latitude) and m.status.status >= 0 and m.latitude != 0.0]
    if not rows:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    t = f.to_sim(np.array([w for w, _ in rows]))
    en = np.array([geo.latlon_to_enu(m.latitude, m.longitude, lat0, lon0) for _, m in rows])
    return t, en[:, 0], en[:, 1]


def error_block(f, t, x, y):
    """Total error in the window, along/cross in cruise."""
    w = f.in_window(t)
    tx, ty = f.truth(t[w])
    out = {'error': error_stats(np.hypot(x[w] - tx, y[w] - ty))}
    c = f.in_cruise(t)
    a, x_ = f.along_cross(t[c], x[c], y[c])
    out['along'] = signed_stats(a)
    out['cross'] = signed_stats(x_)
    return out


def analyze(run_dir, plots=True):
    meta = load_meta(run_dir)
    params = meta.get('parameters', {})
    mission = params.get('guided_mission') or params.get('visual_navigation_planner') or {}
    lat0, lon0 = mission.get('start_lat', LAT0), mission.get('start_lon', LON0)
    route = meta.get('route', {})
    tx = route.get('target_x', mission.get('target_pos_x'))
    ty = route.get('target_y', mission.get('target_pos_y'))

    msgs = read_bag(os.path.join(run_dir, 'bag'))
    f = Flight(msgs)
    f.set_window(msgs)
    grid, win = f.grid, f.window

    s = {'run': meta.get('run', os.path.basename(os.path.normpath(run_dir))),
         'method': meta.get('map', {}).get('label', meta.get('map', {}).get('method')),
         'world': meta.get('world'), 'route': route,
         'meta_reconstructed': bool(meta.get('reconstructed', False))}

    # --- window and timing
    wall_window = np.interp([f.t_takeoff, f.t_end], f.ts, f.tw)
    s['window'] = {'takeoff_s': num(f.t_takeoff - f.ts[0]), 'end_s': num(f.t_end - f.ts[0]),
                   'end': f.end_reason, 'duration_s': num(f.t_end - f.t_takeoff),
                   'cruise_s': num(f.cruise.sum() * DT),
                   'real_time_factor': num((f.t_end - f.t_takeoff) / max(np.diff(wall_window)[0], 1e-6))}

    # --- route: truth vs the planned line, and the outcome at the end of the window
    ekf_t, ekf_x, ekf_y = position_series(f, msgs, '/mavros/global_position/global', lat0, lon0)
    est_t, est_x, est_y = position_series(f, msgs, '/visual_gps', lat0, lon0)
    ekf_gx, ekf_gy = on_grid(grid, ekf_t, ekf_x), on_grid(grid, ekf_t, ekf_y)
    est_gx, est_gy = on_grid(grid, est_t, est_x), on_grid(grid, est_t, est_y)

    have_route = tx is not None and ty is not None and math.hypot(tx, ty) > 1.0
    if have_route:
        length = math.hypot(tx, ty)
        ux, uy = tx / length, ty / length
        along_route = f.x * ux + f.y * uy
        truth_cross = -f.x * uy + f.y * ux
        ekf_cross = -ekf_gx * uy + ekf_gy * ux
    else:
        along_route = truth_cross = ekf_cross = np.full(len(grid), np.nan)
    i_end = min(np.searchsorted(grid, f.t_end), len(grid) - 1)
    if have_route:
        s['route_following'] = {
            'length_m': num(length),
            'truth_cross_track': signed_stats(truth_cross[win]),
            'ekf_cross_track': signed_stats(ekf_cross[win]),
            'progress_at_end_m': num(along_route[i_end]),
            'truth_to_target_at_end_m': num(math.hypot(f.x[i_end] - tx, f.y[i_end] - ty)),
            'ekf_to_target_at_end_m': num(math.hypot(ekf_gx[i_end] - tx, ekf_gy[i_end] - ty)),
            'ekf_error_at_end_m': num(math.hypot(ekf_gx[i_end] - f.x[i_end], ekf_gy[i_end] - f.y[i_end])),
        }
    sc = f.speed[f.cruise]
    s['speed'] = {'cruise_median': num(np.median(sc)) if len(sc) else None,
                  'cruise_std': num(sc.std()) if len(sc) else None,
                  'max': num(f.speed[win].max()) if win.any() else None}

    # --- EKF and published estimate vs truth
    s['ekf'] = error_block(f, ekf_t, ekf_x, ekf_y)
    s['estimate'] = error_block(f, est_t, est_x, est_y)

    # --- localization (per camera frame)
    diags = [json.loads(m.data) for _, m in msgs['/visual_nav/diagnostics']]
    diags = [d for d in diags if 'stamp' in d]
    d_t = np.array([d['stamp'] for d in diags])
    in_w = f.in_window(d_t) if len(d_t) else np.zeros(0, bool)
    frames = [d for d, ok in zip(diags, in_w) if ok]
    fixes = [d for d in frames if 'fix_x' in d]
    accepted = np.array([d['stamp'] for d in frames if d['status'] in ('MATCH', 'MATCH_INIT')])
    loc = {'frames': len(frames)}
    if frames:
        status = {}
        for d in frames:
            status[d['status']] = status.get(d['status'], 0) + 1
        span = max(frames[-1]['stamp'] - frames[0]['stamp'], 1e-6)
        loc.update({
            'rate_hz': num(len(frames) / span),
            'status': status,
            'fix_ratio': num(len(fixes) / len(frames)),
            'accepted_ratio': num(len(accepted) / len(frames)),
            'longest_gap_s': num(np.diff(np.r_[frames[0]['stamp'], accepted, frames[-1]['stamp']]).max()),
            'time_ms': value_stats([d['ms_total'] for d in frames]),
            'inliers_median': num(np.median([d['inliers'] for d in fixes])) if fixes else None,
            'candidates_median': num(np.median([d.get('n_candidates', 0) for d in frames])),
        })
    if fixes:
        ft = np.array([d['stamp'] for d in fixes])
        fx, fy = np.array([d['fix_x'] for d in fixes]), np.array([d['fix_y'] for d in fixes])
        raw = error_block(f, ft, fx, fy)
        c = f.in_cruise(ft)
        a, _ = f.along_cross(ft[c], fx[c], fy[c])
        cruise_min = max(f.cruise.sum() * DT / 60.0, 1e-6)
        raw['jumps_gt_1m_per_min'] = num((np.abs(np.diff(a)) > 1.0).sum() / cruise_min)
        loc['raw_fix'] = raw
    s['localization'] = loc

    # --- GPS_INPUT: sent by the bridge (bag) and received by MAVProxy/ArduPilot (tlog)
    gi = msgs['/mavros/gps_input/gps_input']
    gps = {}
    if gi:
        t = f.to_sim(np.array([w for w, _ in gi]))
        gps['first_s'] = num(t[0] - f.ts[0])
        gps['arm_wait_s'] = num(f.t_takeoff - t[0])
        w = f.in_window(t)
        gps['sent_rate_hz'] = num(w.sum() / max(f.t_end - f.t_takeoff, 1e-6))
    tlog = read_tlog(os.path.join(run_dir, 'mav.tlog'))
    if tlog is not None:
        s['ardupilot'] = tlog['version']
        tg = np.sort(f.to_sim(tlog['gps_input_t']))
        if len(tg) > 2:
            dt = np.diff(tg)
            pre = tg[1:] < f.t_takeoff
            gps['received_dt_median_s'] = num(np.median(dt))
            gps['received_slow_share_before_takeoff'] = num(np.mean(dt[pre] > GPS_UNHEALTHY_DT)) \
                if pre.any() else None
            w = f.in_window(tg[1:])
            gps['received_slow_share_in_flight'] = num(np.mean(dt[w] > GPS_UNHEALTHY_DT)) if w.any() else None
    s['gps_input'] = gps

    # --- outputs
    write_timeseries(run_dir, f, ekf_gx, ekf_gy, est_gx, est_gy, along_route, truth_cross, ekf_cross)
    write_localization(run_dir, f, diags)
    with open(os.path.join(run_dir, 'summary.json'), 'w') as fh:
        json.dump(s, fh, indent=2)
    if plots:
        from gnss_denied_nav.tools import run_plots
        run_plots.make_all(run_dir, s, f, dict(ekf=(ekf_t, ekf_x, ekf_y), est=(est_t, est_x, est_y)),
                           diags, (tx, ty) if have_route else None, along_route, truth_cross, ekf_cross)
    return s


def write_timeseries(run_dir, f, ekf_x, ekf_y, est_x, est_y, along_route, truth_cross, ekf_cross):
    phase = np.where(f.window, 1, np.where(f.grid < f.t_takeoff, 0, 2))
    cols = {
        't': f.grid - f.ts[0], 'phase': phase, 'cruise': f.cruise.astype(int),
        'truth_x': f.x, 'truth_y': f.y, 'truth_z': f.z, 'truth_speed': f.speed,
        'ekf_x': ekf_x, 'ekf_y': ekf_y, 'est_x': est_x, 'est_y': est_y,
        'ekf_error': np.hypot(ekf_x - f.x, ekf_y - f.y), 'est_error': np.hypot(est_x - f.x, est_y - f.y),
        'route_along': along_route, 'truth_cross_track': truth_cross, 'ekf_cross_track': ekf_cross,
    }
    np.savetxt(os.path.join(run_dir, 'timeseries.csv'), np.column_stack(list(cols.values())),
               delimiter=',', header=','.join(cols), comments='', fmt='%.4f')


def write_localization(run_dir, f, diags):
    import csv
    with open(os.path.join(run_dir, 'localization.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t', 'in_window', 'status', 'ms_total', 'n_candidates', 'inliers', 'sigma_m',
                    'ref_index', 'fix_x', 'fix_y', 'truth_x', 'truth_y', 'fix_error'])
        for d in diags:
            t = d['stamp']
            tx, ty = f.truth(t)
            fx, fy = d.get('fix_x'), d.get('fix_y')
            err = math.hypot(fx - tx, fy - ty) if fx is not None else ''
            w.writerow([f'{t - f.ts[0]:.3f}', int(bool(f.in_window(t))), d['status'], d.get('ms_total'),
                        d.get('n_candidates'), d.get('inliers', ''), d.get('sigma_m', ''),
                        d.get('ref_index', ''),
                        '' if fx is None else f'{fx:.3f}', '' if fy is None else f'{fy:.3f}',
                        f'{tx:.3f}', f'{ty:.3f}', '' if err == '' else f'{err:.3f}'])


# ------------------------------------------------------------------ console
COLUMNS = [
    ('method', 'method', '{}'),
    ('ekf.error.rmse', 'EKF RMSE', '{:.2f}'),
    ('ekf.error.p90', 'EKF p90', '{:.2f}'),
    ('estimate.error.rmse', 'est RMSE', '{:.2f}'),
    ('localization.raw_fix.error.median', 'fix med', '{:.2f}'),
    ('route_following.truth_cross_track.rmse', 'xtrack RMSE', '{:.2f}'),
    ('route_following.truth_to_target_at_end_m', 'to target', '{:.1f}'),
    ('localization.accepted_ratio', 'accepted', '{:.3f}'),
    ('localization.time_ms.median', 'ms/frame', '{:.0f}'),
    ('localization.rate_hz', 'frames/s', '{:.1f}'),
    ('window.end', 'end', '{}'),
]


def get(d, path):
    for key in path.split('.'):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def print_table(summaries):
    rows = [[s['run']] + [('—' if get(s, k) is None else fmt.format(get(s, k))) for k, _, fmt in COLUMNS]
            for s in summaries]
    head = ['run'] + [label for _, label, _ in COLUMNS]
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print('  '.join(c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))


def find_runs(all_runs):
    found = sorted(d for d in glob.glob(os.path.join(os.path.expanduser(LOG_DIR), '*/'))
                   if os.path.isdir(os.path.join(d, 'bag')))
    return found if all_runs else found[-1:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='*', help=f'run directories (default: newest in {LOG_DIR})')
    ap.add_argument('--all', action='store_true', help=f'every run with a bag in {LOG_DIR}')
    ap.add_argument('--no-plots', action='store_true', help='skip the PNG figures')
    args = ap.parse_args(argv)

    runs = [os.path.expanduser(r) for r in args.runs] or find_runs(args.all)
    if not runs:
        print(f'No run with a bag found in {LOG_DIR}')
        return 1
    summaries, failed = [], 0
    for run in runs:
        print(f'Analyzing {run} ...', flush=True)
        try:
            summaries.append(analyze(run, plots=not args.no_plots))
        except Exception as e:
            failed += 1
            print(f'  skipped: {e}')
    if summaries:
        print()
        print_table(summaries)
    return 1 if failed and not summaries else 0


if __name__ == '__main__':
    sys.exit(main())

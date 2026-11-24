#!/usr/bin/env python3
"""Analyze a navigation run from its rosbag, in simulation time.

Usage:
    ros2 run gnss_denied_nav analyze_run                      # newest run in ~/gnss_nav_logs
    ros2 run gnss_denied_nav analyze_run ~/gnss_nav_logs/<run>
    ros2 run gnss_denied_nav analyze_run --compare <run> <run> ...

Needs a run recorded with record_bag:=true. Writes <run>/summary.json.

All messages are put on the simulation clock: /odometry carries sim-time
stamps, and its bag receive times give the wall -> sim mapping for topics
stamped with wall time (MAVROS, /visual_gps).

Along/cross errors are relative to the true direction of travel. Only the
"cruise" part of the flight is used: truth speed > 1.5 m/s and height > 20 m.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

from gnss_denied_nav import geo

LAT0, LON0 = -35.3766471, 149.1652375

TOPICS = ['/odometry', '/visual_nav/diagnostics', '/visual_gps', '/visual_velocity_enu',
          '/mavros/local_position/velocity_local', '/mavros/global_position/global',
          '/mavros/gps_input/gps_input', '/mavros/global_position/compass_hdg']


def read_bag(bag_dir):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in TOPICS if t in types]))
    msgs = {t: [] for t in TOPICS}
    classes = {t: get_message(types[t]) for t in TOPICS if t in types}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msgs[topic].append((t_ns * 1e-9, deserialize_message(data, classes[topic])))
    return msgs


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def analyze(run_dir):
    msgs = read_bag(os.path.join(run_dir, 'bag'))
    od = msgs['/odometry']
    if len(od) < 100:
        raise RuntimeError('not enough /odometry in the bag')

    # --- truth and the wall -> sim clock mapping
    tw = np.array([w for w, _ in od])
    ts = np.array([stamp(m) for _, m in od])
    p = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                  for _, m in od])
    p -= p[0]                                   # home = first pose (on the ground)
    # True heading (compass convention) from the odometry orientation.
    quats = [m.pose.pose.orientation for _, m in od]
    yaw = np.unwrap([math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                     for q in quats])
    roll = np.array([math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
                     for q in quats])
    true_hdg = np.degrees(np.pi / 2 - yaw)
    to_sim = lambda w: np.interp(w, tw, ts)     # noqa: E731

    grid = np.arange(ts[0], ts[-1], 0.1)
    gx, gy, gz = (np.interp(grid, ts, p[:, k]) for k in range(3))
    # Truth velocity over 0.5 s (central difference on the 0.1 s grid).
    vx = np.gradient(gx, 0.1)
    vy = np.gradient(gy, 0.1)
    k = 5
    vx = np.convolve(vx, np.ones(k) / k, mode='same')
    vy = np.convolve(vy, np.ones(k) / k, mode='same')
    speed = np.hypot(vx, vy)
    cruise = (speed > 1.5) & (gz > 20.0)

    def truth_at(t):
        return np.interp(t, grid, gx), np.interp(t, grid, gy)

    def dir_at(t):
        ux, uy = np.interp(t, grid, vx), np.interp(t, grid, vy)
        n = np.maximum(np.hypot(ux, uy), 1e-6)
        return ux / n, uy / n

    def in_cruise(t):
        return np.interp(t, grid, cruise.astype(float)) > 0.5

    def along_cross(t, x, y):
        tx, ty = truth_at(t)
        ux, uy = dir_at(t)
        ex, ey = x - tx, y - ty
        return ex * ux + ey * uy, -ex * uy + ey * ux

    s = {'run': os.path.basename(os.path.normpath(run_dir))}
    s['sim_duration_s'] = float(ts[-1] - ts[0])
    s['real_time_factor'] = float((ts[-1] - ts[0]) / (tw[-1] - tw[0]))
    s['cruise_s'] = float(cruise.sum() * 0.1)
    s['distance_m'] = float(np.sum(np.hypot(np.diff(gx), np.diff(gy))))

    # --- speed smoothness (truth): 0.5 s speeds in cruise
    sc = speed[cruise]
    acc = np.abs(np.gradient(speed, 0.1))[cruise]
    groll = np.degrees(np.interp(grid, ts, roll))[cruise]
    # Lateral (cross-track) velocity relative to the 10 s mean direction of travel.
    k10 = 100
    mvx = np.convolve(vx, np.ones(k10) / k10, mode='same')
    mvy = np.convolve(vy, np.ones(k10) / k10, mode='same')
    mn = np.maximum(np.hypot(mvx, mvy), 1e-6)
    lat = (-vx * mvy + vy * mvx) / mn
    s['truth_attitude'] = {'roll_std_deg': float(groll.std()) if len(groll) else float('nan'),
                           'lateral_velocity_rms': float(np.sqrt(np.mean(lat[cruise] ** 2)))}
    s['truth_speed'] = {'median': pct(sc, 50), 'std': float(sc.std()), 'p5': pct(sc, 5),
                        'p95': pct(sc, 95), 'max': float(sc.max()) if len(sc) else float('nan'),
                        'accel_rms': float(np.sqrt(np.mean(acc ** 2))) if len(acc) else float('nan')}

    # --- EKF velocity and position vs truth
    ev = msgs['/mavros/local_position/velocity_local']
    if ev:
        t = to_sim(np.array([w for w, _ in ev]))
        v = np.array([[m.twist.linear.x, m.twist.linear.y] for _, m in ev])
        c = in_cruise(t)
        err = np.hypot(v[c, 0] - np.interp(t[c], grid, vx), v[c, 1] - np.interp(t[c], grid, vy))
        s['ekf_velocity_error'] = {'rms': float(np.sqrt(np.mean(err ** 2))), 'p95': pct(err, 95)}
    eg = msgs['/mavros/global_position/global']
    if eg:
        t = to_sim(np.array([w for w, _ in eg]))
        en = np.array([geo.latlon_to_enu(m.latitude, m.longitude, LAT0, LON0) for _, m in eg])
        c = in_cruise(t)
        a, x = along_cross(t[c], en[c, 0], en[c, 1])
        s['ekf_position_error'] = {'along_mean': float(a.mean()), 'along_std': float(a.std()),
                                   'cross_mean': float(x.mean()), 'cross_std': float(x.std()),
                                   'total_median': pct(np.hypot(a, x), 50)}

    # --- optical flow velocity (image-stamped = sim time)
    fl = msgs['/visual_velocity_enu']
    if fl:
        t = np.array([stamp(m) for _, m in fl])
        v = np.array([[m.twist.linear.x, m.twist.linear.y] for _, m in fl])
        c = in_cruise(t)
        sp = np.hypot(v[c, 0], v[c, 1]) - np.interp(t[c], grid, speed)
        s['flow_speed_error'] = {'mean': float(sp.mean()), 'std': float(sp.std())}
        ee = v[c, 0] - np.interp(t[c], grid, vx)
        en = v[c, 1] - np.interp(t[c], grid, vy)
        s['flow_velocity_error'] = {'east_mean': float(ee.mean()), 'east_std': float(ee.std()),
                                    'north_mean': float(en.mean()), 'north_std': float(en.std())}

    # --- EKF heading (compass_hdg) vs true heading
    hd = msgs['/mavros/global_position/compass_hdg']
    if hd:
        t = to_sim(np.array([w for w, _ in hd]))
        c = in_cruise(t)
        err = (np.array([m.data for _, m in hd])[c] - np.interp(t[c], ts, true_hdg) + 180.0) % 360.0 - 180.0
        s['ekf_heading_error'] = {'mean': float(err.mean()), 'std': float(err.std()),
                                  'max_abs': float(np.abs(err).max()) if len(err) else float('nan')}

    # --- raw map fixes at capture time (diagnostics, image-stamped)
    diags = [json.loads(m.data) for _, m in msgs['/visual_nav/diagnostics']]
    fixes = [d for d in diags if 'fix_x' in d and in_cruise(d['stamp'])]
    frames = [d for d in diags if in_cruise(d['stamp'])]
    if fixes:
        t = np.array([d['stamp'] for d in fixes])
        a, x = along_cross(t, np.array([d['fix_x'] for d in fixes]),
                           np.array([d['fix_y'] for d in fixes]))
        jumps = np.abs(np.diff(a)) > 1.0
        s['raw_fix_error'] = {'along_mean': float(a.mean()), 'along_std': float(a.std()),
                              'cross_mean': float(x.mean()), 'cross_std': float(x.std()),
                              'total_median': pct(np.hypot(a, x), 50),
                              'total_p95': pct(np.hypot(a, x), 95),
                              'jumps_gt_1m_per_min': float(jumps.sum() / (s['cruise_s'] / 60.0))}
        s['localization'] = {
            'rate_hz_sim': len(frames) / s['cruise_s'],
            'fix_ratio': len(fixes) / max(len(frames), 1),
            'accepted_ratio': (sum(d['status'] in ('MATCH', 'MATCH_INIT') for d in frames)
                               / max(len(frames), 1)),
            'latency_ms_median': pct([d['ms_total'] for d in frames], 50),
            'latency_ms_p95': pct([d['ms_total'] for d in frames], 95),
            'inliers_median': pct([d['inliers'] for d in fixes], 50),
        }

    # --- published estimate (what goes to GPS_INPUT)
    vg = msgs['/visual_gps']
    if vg:
        t = to_sim(np.array([w for w, _ in vg]))
        en = np.array([geo.latlon_to_enu(m.latitude, m.longitude, LAT0, LON0) for _, m in vg])
        c = in_cruise(t)
        a, x = along_cross(t[c], en[c, 0], en[c, 1])
        s['published_error'] = {'along_mean': float(a.mean()), 'along_std': float(a.std()),
                                'cross_mean': float(x.mean()), 'cross_std': float(x.std()),
                                'total_median': pct(np.hypot(a, x), 50)}

    # --- GPS_INPUT stream
    gi = msgs['/mavros/gps_input/gps_input']
    if gi:
        t = to_sim(np.array([w for w, _ in gi]))
        c = in_cruise(t)
        sent = [m for (_, m), ok in zip(gi, c) if ok]
        s['gps_input'] = {
            'rate_hz_sim': len(sent) / s['cruise_s'],
            'velocity_share': float(np.mean([(m.ignore_flags & 8) == 0 for m in sent])) if sent else 0.0,
            'horiz_accuracy_median': pct([m.horiz_accuracy for m in sent], 50),
            'speed_accuracy_median': pct([m.speed_accuracy for m in sent if (m.ignore_flags & 32) == 0], 50),
        }
    return s


ROWS = [
    ('truth_speed.std', 'truth speed std (m/s)'), ('truth_speed.p5', 'truth speed p5'),
    ('truth_speed.p95', 'truth speed p95'), ('truth_speed.max', 'truth speed max'),
    ('truth_speed.accel_rms', 'truth accel RMS (m/s2)'),
    ('truth_attitude.roll_std_deg', 'truth roll std (deg)'),
    ('truth_attitude.lateral_velocity_rms', 'truth lateral vel RMS'),
    ('ekf_velocity_error.rms', 'EKF velocity error RMS'),
    ('ekf_position_error.along_std', 'EKF along std (m)'),
    ('ekf_position_error.total_median', 'EKF error median'),
    ('published_error.along_mean', 'published along mean'),
    ('published_error.along_std', 'published along std'),
    ('published_error.cross_std', 'published cross std'),
    ('raw_fix_error.total_median', 'raw fix error median'),
    ('raw_fix_error.jumps_gt_1m_per_min', 'fix jumps >1 m /min'),
    ('flow_speed_error.mean', 'flow speed error mean'), ('flow_speed_error.std', 'flow speed error std'),
    ('flow_velocity_error.east_mean', 'flow east error mean'),
    ('flow_velocity_error.north_mean', 'flow north error mean'),
    ('ekf_heading_error.mean', 'EKF heading error mean (deg)'),
    ('ekf_heading_error.max_abs', 'EKF heading error max (deg)'),
    ('localization.rate_hz_sim', 'localization rate (Hz)'),
    ('localization.latency_ms_median', 'latency median (ms)'),
    ('gps_input.velocity_share', 'GPS_INPUT with velocity'), ('real_time_factor', 'real-time factor'),
]


def get(d, path):
    for key in path.split('.'):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def print_table(summaries):
    # Run names start with the date; the time and world are what tells runs apart.
    names = [s['run'][11:] if len(s['run']) > 11 else s['run'] for s in summaries]
    w = max(12, *(len(n) for n in names))
    print(f"{'':28}" + ''.join(f'{n:>{w + 2}}' for n in names))
    for path, label in ROWS:
        vals = [get(s, path) for s in summaries]
        if all(v is None for v in vals):
            continue
        print(f'{label:28}' + ''.join(
            f"{'—' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.2f}':>{w + 2}}"
            for v in vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='*', help='run directories (default: newest in ~/gnss_nav_logs)')
    ap.add_argument('--compare', action='store_true',
                    help='print the runs side by side (uses summary.json when present)')
    args = ap.parse_args(argv)

    runs = [os.path.expanduser(r) for r in args.runs]
    if not runs:
        found = sorted(d for d in glob.glob(os.path.expanduser('~/gnss_nav_logs/*/'))
                       if os.path.isdir(os.path.join(d, 'bag')))
        if not found:
            print('No run with a bag found in ~/gnss_nav_logs')
            return 1
        runs = [found[-1]]

    summaries = []
    for run in runs:
        path = os.path.join(run, 'summary.json')
        if args.compare and os.path.exists(path):
            with open(path) as f:
                summaries.append(json.load(f))
            continue
        print(f'Analyzing {run} ...', flush=True)
        s = analyze(run)
        with open(path, 'w') as f:
            json.dump(s, f, indent=2)
        summaries.append(s)
    print()
    print_table(summaries)
    return 0


if __name__ == '__main__':
    sys.exit(main())

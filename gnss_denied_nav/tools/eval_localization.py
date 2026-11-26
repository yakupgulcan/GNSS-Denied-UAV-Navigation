#!/usr/bin/env python3
"""Offline localization benchmark against GPS-tagged frames (no ROS, no Gazebo).

Each query frame is localized against the map and compared with its own GPS
tag. A query frame that is itself in the map (same file name) is excluded
from the map (leave-one-out), so the result reflects matching against
neighbouring frames only.

Usage:
    ros2 run gnss_denied_nav eval_localization --db ~/frames_X/map_sift_n2000.npz
    ros2 run gnss_denied_nav eval_localization --db map.npz --queries ~/frames_Y
    ros2 run gnss_denied_nav eval_localization --db map.npz --queries ~/frames_Y --mode global
    ... --candidates 5 --accept-inliers 0 --tag ablation --stride 3

Every evaluation is saved like a flight, in its own directory

    ~/gnss_nav_logs/offline/<time>_<queries>_<method>_<settings>[_<tag>]/
        eval.json     settings, map (method, settings, hash), code state, metrics
        frames.csv    one row per query frame

<queries> is "loo-<survey>" for leave-one-out on the map's own survey,
otherwise the query directory name. <settings> are the localizer settings,
e.g. c3-ai150 (candidates, accept_inliers), plus "global", "nohdg" or
"hn<deg>" when used. --no-save only prints.

Defaults for --candidates and --accept-inliers come from the installed
visual_nav_params.yaml, so offline results use the same settings as flights.

Modes:
    prior   search around the true position plus Gaussian noise (--prior-noise),
            like tracking after the first fix
    global  no prior: voting retrieval over the whole map (first fix)

A heading prior (the frame's heading tag plus --heading-noise) is given in
both modes, as the autopilot's compass heading is in flight. --no-heading
disables it.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np

from gnss_denied_nav import geo
from gnss_denied_nav.localization import CameraModel, Localizer, LocalizerConfig, MapDatabase

OUT_DIR = '~/gnss_nav_logs/offline'
DEFAULT_HEADING_NOISE = 5.0


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float('nan')


def flight_defaults():
    """max_candidates and accept_inliers of the visual estimator in the installed params file."""
    defaults = {'max_candidates': 3, 'accept_inliers': 150}
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('gnss_denied_nav'), 'config',
                            'visual_nav_params.yaml')
        with open(path) as f:
            params = yaml.safe_load(f)['visual_estimator_node']['ros__parameters']
        defaults.update({k: int(params[k]) for k in defaults if k in params})
    except Exception:
        pass
    return defaults


def settings_label(args):
    parts = [f'c{args.candidates}', f'ai{args.accept_inliers or 0}']
    if args.mode == 'global':
        parts.append('global')
    if args.no_heading:
        parts.append('nohdg')
    elif args.heading_noise != DEFAULT_HEADING_NOISE:
        parts.append(f'hn{args.heading_noise:g}')
    return '-'.join(parts)


def save(args, db_path, qdir, leave_one_out, rows, metrics):
    from gnss_denied_nav.run_logging import code_info, map_info
    map_desc = map_info(db_path)
    queries = (f'loo-{os.path.basename(os.path.dirname(os.path.abspath(db_path)))}' if leave_one_out
               else os.path.basename(os.path.normpath(qdir)))
    parts = [datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), queries,
             map_desc.get('label', map_desc['method'].lower()), settings_label(args)]
    if args.tag:
        parts.append(args.tag)
    out = os.path.join(os.path.expanduser(args.out_dir), re.sub(r'[^A-Za-z0-9._-]+', '-', '_'.join(parts)))
    os.makedirs(out, exist_ok=True)
    result = {
        'run': os.path.basename(out),
        'kind': 'offline',
        'started': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'queries': {'dir': os.path.abspath(qdir), 'n': len(rows), 'leave_one_out': leave_one_out,
                    'stride': args.stride},
        'settings': {'mode': args.mode, 'candidates': args.candidates,
                     'accept_inliers': args.accept_inliers, 'heading_prior': not args.no_heading,
                     'heading_noise_deg': args.heading_noise, 'prior_noise_m': args.prior_noise,
                     'radius_m': args.radius, 'hfov_rad': args.hfov, 'seed': args.seed},
        'map': map_desc,
        'code': code_info(),
        'note': args.note,
        'metrics': metrics,
    }
    with open(os.path.join(out, 'eval.json'), 'w') as f:
        json.dump(result, f, indent=2)
    write_csv(os.path.join(out, 'frames.csv'), rows)
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def compute_metrics(rows):
    from gnss_denied_nav.tools.analyze_run import error_stats, value_stats
    ok = [r for r in rows if 'error_m' in r]
    rejects = {}
    for r in rows:
        for item in filter(None, r['rejects'].split(';')):
            k, v = item.split('=')
            rejects[k] = rejects.get(k, 0) + int(v)
    return {
        'queries': len(rows),
        'fixes': len(ok),
        'fix_ratio': round(len(ok) / max(len(rows), 1), 4),
        'error_m': error_stats([r['error_m'] for r in ok]),
        'ref_tag_error_m': error_stats([r['ref_tag_error_m'] for r in ok]),
        'heading_error_deg': error_stats(np.abs([r['heading_err_deg'] for r in ok])),
        'time_ms': value_stats([r['ms'] for r in rows]),
        'candidates_tried_median': float(np.median([r['n_tried'] for r in rows])) if rows else None,
        'rejects': rejects,
    }


def main(argv=None):
    defaults = flight_defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', required=True, help='map database (.npz)')
    ap.add_argument('--queries', help='frames dir with frames_details.json (default: the db directory)')
    ap.add_argument('--mode', choices=['prior', 'global'], default='prior')
    ap.add_argument('--prior-noise', type=float, default=10.0, help='prior position noise, 1-sigma (m)')
    ap.add_argument('--radius', type=float, default=60.0, help='prior search radius (m)')
    ap.add_argument('--heading-noise', type=float, default=DEFAULT_HEADING_NOISE,
                    help='heading prior noise, 1-sigma (deg); the prior is the frame heading plus noise')
    ap.add_argument('--no-heading', action='store_true',
                    help='no heading prior (no query rotation for SuperPoint, no heading check)')
    ap.add_argument('--candidates', type=int, default=defaults['max_candidates'],
                    help=f"map frames matched per query (default {defaults['max_candidates']}, as in flight)")
    ap.add_argument('--accept-inliers', type=int, default=defaults['accept_inliers'],
                    help='stop at the first candidate whose fix has this many inliers; 0 = try all '
                         f"(default {defaults['accept_inliers']}, as in flight)")
    ap.add_argument('--hfov', type=float, default=2.0, help='camera horizontal FOV (rad)')
    ap.add_argument('--stride', type=int, default=1, help='use every n-th query frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=OUT_DIR,
                    help=f'parent of the result directories (default {OUT_DIR})')
    ap.add_argument('--tag', default='', help='label appended to the result directory name')
    ap.add_argument('--note', default='', help='free-text note stored in eval.json')
    ap.add_argument('--no-save', action='store_true', help='print the results only')
    ap.add_argument('--csv', help='also write the per-frame results to this file')
    args = ap.parse_args(argv)

    db_path = os.path.expanduser(args.db)
    db = MapDatabase.load(db_path)
    for note in db.notes:
        print(f'NOTE: {note}')
    origin = (float(db.lat[0]), float(db.lon[0]))
    db.set_origin(*origin)

    qdir = os.path.expanduser(args.queries) if args.queries else os.path.dirname(db_path)
    with open(os.path.join(qdir, 'frames_details.json')) as f:
        entries = [e for e in json.load(f) if (e.get('gps') or {}).get('lat') is not None]
    entries = entries[::args.stride]
    index_of = {str(n): i for i, n in enumerate(db.filenames)}
    leave_one_out = any(e['filename'] in index_of for e in entries)

    w, h = db.image_size
    loc = Localizer(db, CameraModel(w, h, args.hfov),
                    LocalizerConfig(max_candidates=args.candidates,
                                    accept_inliers=args.accept_inliers or None,
                                    global_retrieval=args.mode == 'global'))
    if args.mode == 'global' and db.image_based:
        print(f'{db.method} is image-based: no descriptors for global retrieval; use --mode prior')
        return 1
    rng = np.random.default_rng(args.seed)
    print(f'{db.method} map: {len(db)} frames | {len(entries)} queries from {qdir} | '
          f'mode={args.mode}{" (leave-one-out)" if leave_one_out else ""} | {settings_label(args)}')

    rows = []
    for n, e in enumerate(entries):
        gray = cv2.imread(os.path.join(qdir, e['filename']), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gt = geo.latlon_to_enu(e['gps']['lat'], e['gps']['lon'], *origin)
        exclude = [index_of[e['filename']]] if e['filename'] in index_of else []
        prior = None
        if args.mode == 'prior':
            prior = (gt[0] + rng.normal(0, args.prior_noise), gt[1] + rng.normal(0, args.prior_noise))
        heading = None
        if not args.no_heading and e.get('heading') is not None:
            heading = (e['heading'] + rng.normal(0, args.heading_noise)) % 360.0
        res, info = loc.localize(gray, prior, args.radius, exclude, heading)

        row = {'filename': e['filename'], 'gt_x': gt[0], 'gt_y': gt[1],
               'ms': info['ms_total'], 'n_candidates': len(info['candidates']),
               'n_tried': info.get('n_tried', len(info['candidates'])),
               'rejects': ';'.join(f'{k}={v}' for k, v in info['rejects'].items())}
        if res is not None:
            ref = db.enu[res.ref_index]
            hdg_err = (res.heading_deg - (e.get('heading') or 0.0) + 180.0) % 360.0 - 180.0
            row.update(est_x=res.x, est_y=res.y, error_m=math.hypot(res.x - gt[0], res.y - gt[1]),
                       ref_tag_error_m=math.hypot(ref[0] - gt[0], ref[1] - gt[1]),
                       heading_err_deg=hdg_err, sigma_m=res.sigma_m, inliers=res.n_inliers,
                       scale=res.scale, ref=str(db.filenames[res.ref_index]))
        rows.append(row)
        if (n + 1) % 50 == 0:
            print(f'  {n + 1}/{len(entries)}', flush=True)

    ok = [r for r in rows if 'error_m' in r]
    err = np.array([r['error_m'] for r in ok])
    tag = np.array([r['ref_tag_error_m'] for r in ok])
    hdg = np.abs([r['heading_err_deg'] for r in ok])
    ms = np.array([r['ms'] for r in rows])
    print()
    print(f'fixes:            {len(ok)}/{len(rows)} ({100.0 * len(ok) / max(len(rows), 1):.1f}%)')
    print(f'position error:   median {pct(err, 50):.2f} m | p90 {pct(err, 90):.2f} m | '
          f'max {err.max() if len(err) else float("nan"):.2f} m')
    print(f'ref-frame tag:    median {pct(tag, 50):.2f} m | p90 {pct(tag, 90):.2f} m '
          f'(error if the matched frame\'s GPS tag were used as the position)')
    print(f'heading error:    median {pct(hdg, 50):.2f} deg | p90 {pct(hdg, 90):.2f} deg')
    print(f'latency:          median {pct(ms, 50):.1f} ms | p90 {pct(ms, 90):.1f} ms')

    if args.csv:
        write_csv(args.csv, rows)
        print(f'per-frame results -> {args.csv}')
    if not args.no_save:
        out = save(args, db_path, qdir, leave_one_out, rows, compute_metrics(rows))
        print(f'saved -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

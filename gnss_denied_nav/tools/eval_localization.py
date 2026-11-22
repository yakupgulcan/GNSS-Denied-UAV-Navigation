#!/usr/bin/env python3
"""Offline localization benchmark against GPS-tagged frames (no ROS, no Gazebo).

Each query frame is localized against the map and compared with its own GPS
tag. A query frame that is itself in the map (same file name) is excluded
from the map (leave-one-out), so the result reflects matching against
neighbouring frames only.

Usage:
    ros2 run gnss_denied_nav eval_localization --db ~/frames_X/map_sift.npz
    ros2 run gnss_denied_nav eval_localization --db map.npz --queries ~/frames_Y --mode global
    ... --csv results.csv --stride 3

Modes:
    prior   search around the true position plus Gaussian noise (--prior-noise),
            like tracking after the first fix
    global  no prior: voting retrieval over the whole map (first fix)
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

from gnss_denied_nav import geo
from gnss_denied_nav.localization import CameraModel, Localizer, LocalizerConfig, MapDatabase


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float('nan')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', required=True, help='map database (.npz)')
    ap.add_argument('--queries', help='frames dir with frames_details.json (default: the db directory)')
    ap.add_argument('--mode', choices=['prior', 'global'], default='prior')
    ap.add_argument('--prior-noise', type=float, default=10.0, help='prior position noise, 1-sigma (m)')
    ap.add_argument('--radius', type=float, default=60.0, help='prior search radius (m)')
    ap.add_argument('--candidates', type=int, default=5)
    ap.add_argument('--hfov', type=float, default=2.0, help='camera horizontal FOV (rad)')
    ap.add_argument('--stride', type=int, default=1, help='use every n-th query frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv', help='write per-frame results here')
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
                                    global_retrieval=args.mode == 'global'))
    rng = np.random.default_rng(args.seed)
    print(f'{db.method} map: {len(db)} frames | {len(entries)} queries from {qdir} | '
          f'mode={args.mode}{" (leave-one-out)" if leave_one_out else ""}')

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
        res, info = loc.localize(gray, prior, args.radius, exclude)

        row = {'filename': e['filename'], 'gt_x': gt[0], 'gt_y': gt[1],
               'ms': info['ms_total'], 'n_candidates': len(info['candidates']),
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
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f'per-frame results -> {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Build a map database from a survey recording (save_frames output).

Usage:
    ros2 run gnss_denied_nav build_map_db ~/frames_<stamp> --method SIFT
    ros2 run gnss_denied_nav build_map_db ~/frames_<stamp> --method ORB --n-features 3000

Writes <frames_dir>/map_<method>.npz unless --out is given.
"""
import argparse
import os
import sys
import time

from gnss_denied_nav.localization.database import MapDatabase
from gnss_denied_nav.localization.features import EXTRACTORS, create_extractor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames_dir', help='directory with frames_details.json and images')
    ap.add_argument('--method', default='SIFT', choices=sorted(EXTRACTORS))
    ap.add_argument('--n-features', type=int, help='SIFT/ORB feature count')
    ap.add_argument('--threshold', type=float, help='AKAZE/BRISK detector threshold')
    ap.add_argument('--out', help='output .npz path')
    args = ap.parse_args(argv)

    frames_dir = os.path.expanduser(args.frames_dir)
    params = {}
    if args.n_features is not None:
        params['n_features'] = args.n_features
    if args.threshold is not None:
        params['threshold'] = args.threshold
    extractor = create_extractor(args.method, **params)
    out = os.path.expanduser(args.out) if args.out else os.path.join(
        frames_dir, f'map_{extractor.name.lower()}.npz')

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f'  {done}/{total} frames', flush=True)

    print(f'Building {extractor.name} map {extractor.params} from {frames_dir}')
    t0 = time.time()
    db = MapDatabase.build(frames_dir, extractor, progress)
    db.save(out)
    print(f'Saved {len(db)} frames, {len(db.kpts)} features '
          f'({time.time() - t0:.1f} s) -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

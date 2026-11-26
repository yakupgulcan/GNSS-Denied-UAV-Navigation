"""
Per-run log directories.

Every navigation launch creates one run directory,

    <log_dir>/<YYYY-MM-DD_HH-MM-SS>_<world>_<method>_<route>[_<tag>]/
    e.g. 2026-11-26_14-02-11_map_changed_sift-n2000_1km

and everything a run produces goes there:

    meta.json                            what was flown (see describe_run)
    *.yaml, *.param                      parameter files as used
    mav.tlog, eeprom.bin, logs/*.BIN     ArduPilot SITL / MAVProxy (launch cwd)
    bag/                                 rosbag of the navigation topics: the
                                         run record, read by analyze_run

ArduPilot SITL and MAVProxy write into their working directory, so the
launch process changes into the run directory before starting them. This
also gives every run a fresh SITL eeprom (parameters from the .param files
only, nothing left over from earlier runs).
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

DEFAULT_LOG_DIR = '~/gnss_nav_logs'

# /clock is left out on purpose: at ~800 Hz it was ~90% of the bag, and every
# recorded message carries its own (simulation) header stamp anyway.
BAG_TOPICS = [
    '/odometry',
    '/visual_gps', '/visual_pose_enu',
    '/visual_nav/diagnostics', '/visual_nav/map_frames',
    '/visual_velocity_enu', '/visual_velocity_body',
    '/mavros/state', '/mavros/global_position/global', '/mavros/global_position/rel_alt',
    '/mavros/global_position/compass_hdg', '/mavros/local_position/pose',
    '/mavros/local_position/velocity_local', '/mavros/gps_input/gps_input',
    '/mavros/setpoint_raw/global', '/mavros/setpoint_raw/target_global',
]


ESTIMATOR_NODE = 'visual_estimator_node'


def launch_overrides(cfg, mission_node='guided_mission'):
    """Node parameters set by launch arguments, by node name.

    The launch file passes these to the nodes and describe_run records them,
    so meta.json shows exactly what the nodes were given. Empty arguments
    leave the params_file value.
    """
    out = {ESTIMATOR_NODE: {}, mission_node: {}}
    db_path = (cfg.get('db_path') or '').strip()
    if db_path:
        out[ESTIMATOR_NODE]['db_path'] = db_path
    for arg, param in (('target_x', 'target_pos_x'), ('target_y', 'target_pos_y')):
        value = (cfg.get(arg) or '').strip()
        if value:
            out[mission_node][param] = float(value)
    return out


def _yaml_params(params_file):
    import yaml
    if not params_file or not os.path.isfile(params_file):
        return {}
    with open(params_file) as f:
        data = yaml.safe_load(f) or {}
    return {node: dict(v.get('ros__parameters', {})) for node, v in data.items() if isinstance(v, dict)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _map_info(db_path):
    """Method and settings of the map database (read from its own metadata)."""
    info = {'db_path': db_path, 'method': 'unknown', 'params': {}}
    path = os.path.expanduser(db_path or '')
    if not os.path.isfile(path):
        info['error'] = 'database file not found'
        return info
    try:
        from gnss_denied_nav.localization.database import MapDatabase, method_tags
        meta = MapDatabase.read_meta(path)
    except Exception as e:  # unreadable file: still start the run, but say so
        info['error'] = f'could not read database metadata: {e}'
        return info
    if meta is None:
        info['method'] = 'legacy'
        info['label'] = 'legacy'
    else:
        info.update(method=meta['method'], params=meta['params'],
                    label='-'.join(method_tags(meta['method'], meta['params'])),
                    image_based=bool(meta.get('image_based', False)),
                    created=meta.get('created'))
    info['db_size_bytes'] = os.path.getsize(path)
    info['db_sha256'] = _sha256(path)
    return info


def _source_dir():
    """Source checkout of this package, found next to its install space (None if not found)."""
    try:
        from ament_index_python.packages import get_package_prefix
        workspace = os.path.dirname(os.path.dirname(get_package_prefix('gnss_denied_nav')))
    except Exception:
        return None
    for root, dirs, files in os.walk(os.path.join(workspace, 'src')):
        if root.count(os.sep) - workspace.count(os.sep) > 3:
            dirs[:] = []
            continue
        if 'package.xml' in files:
            with open(os.path.join(root, 'package.xml')) as f:
                if '<name>gnss_denied_nav</name>' in f.read():
                    return root
            dirs[:] = []
    return None


def _git(src, *args):
    return subprocess.run(['git', '-C', src, *args], capture_output=True, text=True,
                          timeout=10).stdout


def _code_info():
    """Git state of the source, and source files that differ from the installed copy.

    ``ros2 launch``/``ros2 run`` use the installed copy: a non-empty
    'not_installed' list means the run did not use the code at 'commit'.
    """
    src = _source_dir()
    if src is None:
        return {'source_dir': None}
    info = {'source_dir': src}
    try:
        info['commit'] = _git(src, 'rev-parse', 'HEAD').strip()
        info['branch'] = _git(src, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
        info['uncommitted'] = [line[3:] for line in _git(src, 'status', '--porcelain',
                                                         '--untracked-files=no').splitlines()]
    except Exception as e:
        info['git_error'] = str(e)

    import filecmp
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    prefix = get_package_prefix('gnss_denied_nav')
    py_dir = os.path.join(prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
                          'site-packages', 'gnss_denied_nav')
    share = get_package_share_directory('gnss_denied_nav')
    pairs = []
    for root, _, files in os.walk(os.path.join(src, 'gnss_denied_nav')):
        for name in files:
            if name.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, name), os.path.join(src, 'gnss_denied_nav'))
                pairs.append((os.path.join('gnss_denied_nav', rel), os.path.join(py_dir, rel)))
    for sub in ('launch', 'config'):
        for root, _, files in os.walk(os.path.join(src, sub)):
            if '__pycache__' in root:
                continue
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), src)
                pairs.append((rel, os.path.join(share, rel)))
    info['not_installed'] = sorted(
        rel for rel, installed in pairs
        if not (os.path.isfile(installed) and filecmp.cmp(os.path.join(src, rel), installed, shallow=False)))
    return info


def _route_label(distance_m):
    if distance_m is None:
        return 'route'
    if distance_m >= 1000:
        return f'{distance_m / 1000:.3g}km'
    return f'{distance_m:.0f}m'


def describe_run(cfg, pipeline, mission_node):
    """Everything needed to identify and reproduce a run (written to meta.json)."""
    params = _yaml_params(cfg.get('params_file', ''))
    for node, values in launch_overrides(cfg, mission_node).items():
        params.setdefault(node, {}).update(values)
    estimator = params.get(ESTIMATOR_NODE, {})
    mission = params.get(mission_node, {})

    tx, ty = mission.get('target_pos_x'), mission.get('target_pos_y')
    distance = math.hypot(tx, ty) if tx is not None and ty is not None else None
    world = os.path.splitext(os.path.basename(cfg.get('world', 'unknown')))[0]
    return {
        'started': datetime.now().isoformat(timespec='seconds'),
        'pipeline': pipeline,
        'world': world,
        'map': _map_info(estimator.get('db_path')),
        'route': {'target_x': tx, 'target_y': ty, 'distance_m': distance,
                  'altitude_m': mission.get('target_altitude')},
        'tag': (cfg.get('tag') or '').strip(),
        'note': (cfg.get('note') or '').strip(),
        'code': _code_info(),
        'parameters': params,
        'command': ' '.join(sys.argv),
        'launch_arguments': dict(cfg),
    }


def run_name(meta, stamp):
    """<stamp>_<world>_<method>_<route>[_<tag>], filesystem-safe."""
    parts = [stamp, meta['world'], meta['map'].get('label', meta['map']['method'].lower()),
             _route_label(meta['route']['distance_m'])]
    if meta['tag']:
        parts.append(meta['tag'])
    return re.sub(r'[^A-Za-z0-9._-]+', '-', '_'.join(parts))


def run_logging_actions(files_to_copy=(), pipeline='guided', mission_node='guided_mission'):
    """Launch actions that set up the run directory.

    Declares the launch arguments log_dir, record_bag, bag_images, tag and
    note, creates the run directory with meta.json, changes into it, sets the
    launch configuration 'run_dir' and starts rosbag (unless record_bag:=false).
    Place the returned actions before the simulation include.
    """
    from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
                                SetLaunchConfiguration)

    def setup(context):
        cfg = context.launch_configurations
        meta = describe_run(cfg, pipeline, mission_node)
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = os.path.join(os.path.expanduser(cfg['log_dir']), run_name(meta, stamp))
        meta['run'] = os.path.basename(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)

        for path in [*files_to_copy, cfg.get('params_file', '')]:
            if path and os.path.isfile(path):
                shutil.copy(path, run_dir)
        with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        actions = [SetLaunchConfiguration('run_dir', run_dir)]
        if cfg.get('record_bag', 'true').lower() == 'true':
            topics = BAG_TOPICS + (['/camera/image']
                                   if cfg.get('bag_images', 'false').lower() == 'true' else [])
            actions.append(ExecuteProcess(
                cmd=['ros2', 'bag', 'record', '-o', os.path.join(run_dir, 'bag'), *topics],
                output='log'))
        print(f'[run_logging] run directory: {run_dir}')
        if meta['map'].get('error'):
            print(f"[run_logging] WARNING: {meta['map']['error']}: {meta['map']['db_path']}")
        if meta['code'].get('not_installed'):
            print('[run_logging] WARNING: source files differ from the installed copy '
                  f"(rebuild?): {', '.join(meta['code']['not_installed'])}")
        return actions

    return [
        DeclareLaunchArgument('log_dir', default_value=DEFAULT_LOG_DIR,
                              description='Parent directory of the per-run log directories'),
        DeclareLaunchArgument('record_bag', default_value='true',
                              description='Record a rosbag of the navigation topics '
                                          '(the run record that analyze_run reads)'),
        DeclareLaunchArgument('bag_images', default_value='false',
                              description='Also record /camera/image in the rosbag (large)'),
        DeclareLaunchArgument('tag', default_value='',
                              description='Optional label appended to the run directory name'),
        DeclareLaunchArgument('note', default_value='',
                              description='Optional free-text note stored in meta.json'),
        OpaqueFunction(function=setup),
    ]

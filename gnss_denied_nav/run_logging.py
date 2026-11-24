"""
Per-run log directories.

Every navigation launch creates one run directory,

    <log_dir>/<YYYY-MM-DD_HH-MM-SS>_<world>/

and everything a run produces goes there:

    perf_metrics.csv, visual_log.csv     our loggers (run_dir parameter)
    run_meta.json, *.yaml, *.param       launch arguments and parameter files
    mav.tlog, eeprom.bin, logs/*.BIN     ArduPilot SITL / MAVProxy (launch cwd)
    bag/                                 rosbag (record_bag:=true)

ArduPilot SITL and MAVProxy write into their working directory, so the
launch process changes into the run directory before starting them. This
also gives every run a fresh SITL eeprom (parameters from the .param files
only, nothing left over from earlier runs).
"""
import json
import os
import shutil
import sys
from datetime import datetime

DEFAULT_LOG_DIR = '~/gnss_nav_logs'

BAG_TOPICS = [
    '/clock', '/odometry',
    '/visual_gps', '/visual_pose_enu', '/visual_perf_metrics',
    '/visual_nav/diagnostics', '/visual_nav/map_frames',
    '/visual_velocity_enu', '/visual_velocity_body',
    '/mavros/state', '/mavros/global_position/global', '/mavros/global_position/rel_alt',
    '/mavros/global_position/compass_hdg', '/mavros/local_position/pose',
    '/mavros/local_position/velocity_local', '/mavros/gps_input/gps_input',
    '/mavros/setpoint_raw/global', '/mavros/setpoint_raw/target_global',
]


def resolve_run_dir(run_dir=''):
    """Run directory for a node: the given one, or a fresh one under DEFAULT_LOG_DIR."""
    if not run_dir:
        run_dir = os.path.join(DEFAULT_LOG_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    run_dir = os.path.expanduser(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_logging_actions(files_to_copy=()):
    """Launch actions that set up the run directory.

    Declares the launch arguments log_dir, record_bag and bag_images, creates
    the run directory, changes into it, sets the launch configuration
    'run_dir' (pass it to logging nodes) and optionally starts rosbag.
    Place the returned actions before the simulation include.
    """
    from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
                                SetLaunchConfiguration)

    def setup(context):
        cfg = context.launch_configurations
        world = os.path.splitext(os.path.basename(cfg.get('world', 'run')))[0]
        name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{world}"
        run_dir = os.path.join(os.path.expanduser(cfg['log_dir']), name)
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)

        for path in [*files_to_copy, cfg.get('params_file', '')]:
            if path and os.path.isfile(path):
                shutil.copy(path, run_dir)
        with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
            json.dump({'started': datetime.now().isoformat(timespec='seconds'),
                       'command': ' '.join(sys.argv),
                       'launch_arguments': dict(cfg)}, f, indent=2)

        actions = [SetLaunchConfiguration('run_dir', run_dir)]
        if cfg.get('record_bag', 'false').lower() == 'true':
            topics = BAG_TOPICS + (['/camera/image']
                                   if cfg.get('bag_images', 'false').lower() == 'true' else [])
            actions.append(ExecuteProcess(
                cmd=['ros2', 'bag', 'record', '-o', os.path.join(run_dir, 'bag'), *topics],
                output='log'))
        print(f'[run_logging] run directory: {run_dir}')
        return actions

    return [
        DeclareLaunchArgument('log_dir', default_value=DEFAULT_LOG_DIR,
                              description='Parent directory of the per-run log directories'),
        DeclareLaunchArgument('record_bag', default_value='false',
                              description='Record a rosbag of navigation topics into the run directory'),
        DeclareLaunchArgument('bag_images', default_value='false',
                              description='Also record /camera/image in the rosbag (large)'),
        OpaqueFunction(function=setup),
    ]

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from gnss_denied_nav.run_logging import run_logging_actions


def generate_launch_description():
    pkg_gnss_denied_nav = get_package_share_directory("gnss_denied_nav")

    default_params_file = os.path.join(
        pkg_gnss_denied_nav,
        "config",
        "visual_nav_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to YAML file containing visual navigation parameters",
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="map_changed.sdf",
        description="Gazebo world file for GNSS-denied test flight (default: map_changed.sdf)",
    )

    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Start the Gazebo GUI",
    )

    enable_logger_arg = DeclareLaunchArgument(
        "enable_logger",
        default_value="true",
        description="Automatically record visual performance metrics and ground-truth errors to CSV",
    )

    gps_velocity_arg = DeclareLaunchArgument(
        "gps_velocity",
        default_value="false",
        description="Send optical-flow velocity in GPS_INPUT and let the EKF use it "
                    "(gnss_denied_fake_gps_vel.param, EK3_SRC1_VELXY=3)",
    )

    def select_scenario_params(context):
        use_velocity = LaunchConfiguration("gps_velocity").perform(context).lower() == "true"
        name = "gnss_denied_fake_gps_vel.param" if use_velocity else "gnss_denied_fake_gps.param"
        path = os.path.join(pkg_gnss_denied_nav, "config", "arducopter_params", name)
        return [SetLaunchConfiguration("scenario_params", path)]

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("gnss_denied_nav"),
                        "launch",
                        "simulation.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={
            "scenario_params": LaunchConfiguration("scenario_params"),
            "world": LaunchConfiguration("world"),
            "gui": LaunchConfiguration("gui"),
        }.items(),
    )

    optical_flow = Node(
        package="gnss_denied_nav",
        executable="optical_flow",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    visual_estimator = Node(
        package="gnss_denied_nav",
        executable="visual_estimator",
        parameters=[
            LaunchConfiguration("params_file"),
            # The autopilot needs a valid position from the ground: start at
            # the known home position.
            {"init_at_home": True, "run_dir": LaunchConfiguration("run_dir")},
        ],
        output="screen",
    )

    gps_input_bridge = Node(
        package="gnss_denied_nav",
        executable="gps_input_bridge",
        parameters=[
            LaunchConfiguration("params_file"),
            {"send_velocity": ParameterValue(LaunchConfiguration("gps_velocity"), value_type=bool)},
        ],
        output="screen",
    )

    target_x_arg = DeclareLaunchArgument(
        "target_x", default_value="",
        description="Target east (m); empty = target_pos_x from params_file")
    target_y_arg = DeclareLaunchArgument(
        "target_y", default_value="",
        description="Target north (m), e.g. 1000 for short test flights; empty = target_pos_y from params_file")

    def make_guided_mission(context):
        # Launch arguments override the target in params_file only when given.
        overrides = {}
        for arg, param in (("target_x", "target_pos_x"), ("target_y", "target_pos_y")):
            value = LaunchConfiguration(arg).perform(context).strip()
            if value:
                overrides[param] = float(value)
        return [Node(
            package="gnss_denied_nav",
            executable="guided_mission",
            parameters=[LaunchConfiguration("params_file"), overrides],
            output="screen",
        )]

    perf_metrics_logger = Node(
        package="gnss_denied_nav",
        executable="perf_metrics_logger",
        parameters=[{"run_dir": LaunchConfiguration("run_dir")}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_logger")),
    )

    return LaunchDescription([
        params_file_arg,
        world_arg,
        enable_logger_arg,
        gui_arg,
        # Run directory under log_dir; SITL/MAVProxy files land there too.
        gps_velocity_arg,
        target_x_arg,
        target_y_arg,
        OpaqueFunction(function=select_scenario_params),
        *run_logging_actions(),
        simulation,

        TimerAction(
            period=63.0,
            actions=[optical_flow],
        ),

        TimerAction(
            period=66.0,
            actions=[visual_estimator, perf_metrics_logger],
        ),

        TimerAction(
            period=69.0,
            actions=[gps_input_bridge],
        ),

        # The mission node waits for the EKF position itself before arming.
        TimerAction(
            period=72.0,
            actions=[OpaqueFunction(function=make_guided_mission)],
        ),
    ])
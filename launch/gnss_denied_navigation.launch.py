import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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

    scenario_params = os.path.join(
        pkg_gnss_denied_nav,
        "config",
        "arducopter_params",
        "gnss_denied.param",
    )

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
            "scenario_params": scenario_params,
        }.items(),
    )

    nav_gcs = Node(
        package="gnss_denied_nav",
        executable="nav_gcs",
        output="screen",
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
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    base_controller = Node(
        package="gnss_denied_nav",
        executable="base_controller",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    navigation_runner = Node(
        package="gnss_denied_nav",
        executable="navigation_runner",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    return LaunchDescription([
        params_file_arg,
        simulation,

        TimerAction(
            period=60.0,
            actions=[nav_gcs],
        ),

        TimerAction(
            period=63.0,
            actions=[optical_flow],
        ),

        TimerAction(
            period=66.0,
            actions=[visual_estimator],
        ),

        TimerAction(
            period=69.0,
            actions=[base_controller],
        ),

        TimerAction(
            period=72.0,
            actions=[navigation_runner],
        ),
    ])
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

    # Launch arguments for survey grid parameters
    home_lat_arg = DeclareLaunchArgument(
        "home_lat",
        default_value="-35.3658674",
        description="Reference home latitude for mission origin",
    )
    home_lon_arg = DeclareLaunchArgument(
        "home_lon",
        default_value="149.1652376",
        description="Reference home longitude for mission origin",
    )
    flight_alt_arg = DeclareLaunchArgument(
        "flight_alt",
        default_value="30.0",
        description="Survey flight altitude in meters",
    )
    width_left_arg = DeclareLaunchArgument(
        "width_left",
        default_value="40.0",
        description="Survey area width to the left (West) in meters",
    )
    width_right_arg = DeclareLaunchArgument(
        "width_right",
        default_value="40.0",
        description="Survey area width to the right (East) in meters",
    )
    forward_dist_arg = DeclareLaunchArgument(
        "forward_distance",
        default_value="600.0",
        description="Survey area distance forward (North) in meters",
    )
    backward_dist_arg = DeclareLaunchArgument(
        "backward_distance",
        default_value="0.0",
        description="Survey area distance backward (South) in meters",
    )
    corridor_spacing_arg = DeclareLaunchArgument(
        "corridor_spacing",
        default_value="20.0",
        description="Spacing between parallel survey corridors in meters",
    )

    scenario_params = os.path.join(
        pkg_gnss_denied_nav,
        "config",
        "arducopter_params",
        "gnss_available.param",
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

    save_frames = Node(
        package="gnss_denied_nav",
        executable="save_frames",
        output="screen",
    )

    follow_local_wp = Node(
        package="gnss_denied_nav",
        executable="follow_local_wp",
        output="screen",
        parameters=[{
            "home_lat": LaunchConfiguration("home_lat"),
            "home_lon": LaunchConfiguration("home_lon"),
            "flight_alt": LaunchConfiguration("flight_alt"),
            "width_left": LaunchConfiguration("width_left"),
            "width_right": LaunchConfiguration("width_right"),
            "forward_distance": LaunchConfiguration("forward_distance"),
            "backward_distance": LaunchConfiguration("backward_distance"),
            "corridor_spacing": LaunchConfiguration("corridor_spacing"),
        }],
    )

    return LaunchDescription([
        home_lat_arg,
        home_lon_arg,
        flight_alt_arg,
        width_left_arg,
        width_right_arg,
        forward_dist_arg,
        backward_dist_arg,
        corridor_spacing_arg,

        simulation,

        # Wait for Gazebo, ArduPilot SITL and MAVROS to fully initialize
        TimerAction(
            period=60.0,
            actions=[save_frames],
        ),

        TimerAction(
            period=63.0,
            actions=[follow_local_wp],
        ),
    ])
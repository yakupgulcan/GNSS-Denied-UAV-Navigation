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

    # Launch arguments for corridor survey parameters
    target_x_arg = DeclareLaunchArgument(
        "target_x",
        default_value="0.0",
        description="Target corridor X coordinate in meters",
    )
    target_y_arg = DeclareLaunchArgument(
        "target_y",
        default_value="3000.0",
        description="Target corridor Y distance in meters",
    )
    corridor_width_arg = DeclareLaunchArgument(
        "corridor_width",
        default_value="80.0",
        description="Total swath corridor width in meters (e.g. 80.0 for +-40m buffer; 0 for single line)",
    )
    lane_spacing_arg = DeclareLaunchArgument(
        "lane_spacing",
        default_value="20.0",
        description="Spacing between parallel survey lines in meters",
    )
    flight_alt_arg = DeclareLaunchArgument(
        "flight_alt",
        default_value="30.0",
        description="Survey flight altitude in meters",
    )
    auto_home_arg = DeclareLaunchArgument(
        "auto_home",
        default_value="true",
        description="Automatically capture actual Home GPS coordinates from MAVROS on takeoff",
    )
    home_lat_arg = DeclareLaunchArgument(
        "home_lat",
        default_value="-35.3766471",
        description="Fallback home latitude for mission origin (map_original 3000m)",
    )
    home_lon_arg = DeclareLaunchArgument(
        "home_lon",
        default_value="149.1652374",
        description="Fallback home longitude for mission origin (map_original 3000m)",
    )
    min_record_alt_arg = DeclareLaunchArgument(
        "min_record_alt",
        default_value="25.0",
        description="Minimum altitude in meters before image saving begins",
    )
    save_interval_arg = DeclareLaunchArgument(
        "save_interval",
        default_value="1.5",
        description="Interval in seconds between captured frames",
    )
    rtl_on_finish_arg = DeclareLaunchArgument(
        "rtl_on_finish",
        default_value="true",
        description="Automatically trigger RTL and land when survey and return flight are complete",
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
        parameters=[{
            "min_record_alt": LaunchConfiguration("min_record_alt"),
            "save_interval": LaunchConfiguration("save_interval"),
        }],
    )

    follow_local_wp = Node(
        package="gnss_denied_nav",
        executable="follow_local_wp",
        output="screen",
        parameters=[{
            "target_x": LaunchConfiguration("target_x"),
            "target_y": LaunchConfiguration("target_y"),
            "corridor_width": LaunchConfiguration("corridor_width"),
            "lane_spacing": LaunchConfiguration("lane_spacing"),
            "flight_alt": LaunchConfiguration("flight_alt"),
            "auto_home": LaunchConfiguration("auto_home"),
            "home_lat": LaunchConfiguration("home_lat"),
            "home_lon": LaunchConfiguration("home_lon"),
            "rtl_on_finish": LaunchConfiguration("rtl_on_finish"),
        }],
    )

    return LaunchDescription([
        target_x_arg,
        target_y_arg,
        corridor_width_arg,
        lane_spacing_arg,
        flight_alt_arg,
        auto_home_arg,
        home_lat_arg,
        home_lon_arg,
        min_record_alt_arg,
        save_interval_arg,
        rtl_on_finish_arg,

        simulation,

        # Wait for Gazebo, ArduPilot SITL and MAVROS to fully initialize
        TimerAction(
            period=70.0,
            actions=[save_frames],
        ),

        TimerAction(
            period=73.0,
            actions=[follow_local_wp],
        ),
    ])

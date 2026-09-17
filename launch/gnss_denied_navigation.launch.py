import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_gnss_denied_nav = get_package_share_directory("gnss_denied_nav")

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

    return LaunchDescription([
        simulation,
    ])
# Copyright 2023 ArduPilot.org.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Adapted from https://github.com/gazebosim/ros_gz_project_template
#
# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch an iris quadcopter in Gazebo."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.substitutions import PythonExpression
import os

def generate_launch_description():
    """Generate a launch description for an Iris quadcopter."""
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_ardupilot_sitl = get_package_share_directory("ardupilot_sitl")
    pkg_ardupilot_gazebo = get_package_share_directory("ardupilot_gazebo")
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    scenario_params = LaunchConfiguration("scenario_params")
    #defaults = LaunchConfiguration("defaults")
    base_defaults = (
        os.path.join(
            pkg_ardupilot_gazebo,
            "config",
            "gazebo-iris-gimbal.parm",
        )
        + ","
        + os.path.join(
            pkg_ardupilot_sitl,
            "config",
            "default_params",
            "dds_udp.parm",
        )
    )
    defaults = PythonExpression([
        "'",
        base_defaults,
        ",",
        scenario_params,
        "'",
    ])

    world_path = PathJoinSubstitution(
        [
            FindPackageShare("gnss_denied_nav"),
            "worlds",
            world,
        ]
    )

    # Iris.
    iris = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("gnss_denied_nav"),
                        "launch",
                        "iris_custom.launch.py",
                    ]
                ),
            ]
        ),
        launch_arguments={
            "defaults": defaults,
        }.items(),
    )

    # Gazebo.
    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py"}'
        ),
        launch_arguments={
            "gz_args": ["-v4 -s -r ", world_path]
        }.items(),
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{Path(pkg_ros_gz_sim) / "launch" / "gz_sim.launch.py"}'
        ),
        launch_arguments={"gz_args": "-v4 -g"}.items(),
        condition=IfCondition(gui),
    )

    # MAVROS
    mavros = Node(
        package="mavros",
        executable="mavros_node",
        output="screen",
        parameters=[
            {
                "fcu_url": "udp://127.0.0.1:14550@14555",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario_params",
                default_value="",
                description="Scenario-specific ArduPilot parameter file.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="map_original.sdf",
                description="Gazebo world file.",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start the Gazebo GUI.",
            ),
            gz_sim_server,
            gz_sim_gui,
            iris,
            TimerAction(
                period=50.0,
                actions=[mavros],
            )
        ]
    )
# SITL Simulation Setup Guide

This guide details the setup and execution of the simulation architecture, which includes **ArduPilot SITL**, **ROS 2 Humble**, **MAVROS**, and **Gazebo Harmonic**. The software architecture is identical for both SITL (Software-In-The-Loop) environments and HITL (Hardware-In-The-Loop) or physical real-world setups.

---

## 1. System Components

#### ArduPilot SITL 
ArduPilot is an open-source flight control softwar. SITL allows us to work with ArduPilot without any special hardware. It receives simulated sensor data from Gazebo and computes the drone's behavior before deploying the code to a physical flight controller.

#### ROS 2 (Robot Operating System)
ROS 2 provides a robust framework for developing robot applications. It manages the communication between different software nodes using a publisher-subscriber and service-client architecture.

#### MAVROS
MAVROS is a ROS 2 package that translates between the ArduPilot communication protocol MAVLink and ROS 2 topics/services. This allows our custom ROS 2 navigation nodes to get sensor data from ArduPilot and send flight commands to the ArduPilot flight controller easily.

#### Gazebo Harmonic
Gazebo is a robust 3D physics simulator. It hosts the simulation world, drone models, cameras, lidars, and other virtual sensors. Thanks to its plugins, Gazebo communicates bi-directionally with ROS 2 and ArduPilot to provide highly accurate simulation data.

---

## 2. Installation Steps

>**Prerequisite:** A machine running **Ubuntu 22.04 LTS**.

### 2.1 ROS 2 Humble Installation
Install the `ros-humble-desktop` package by following the official documentation:
* [ROS 2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

### 2.2 ArduPilot SITL Installation

**1. Install Dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-vcstool python3-rosdep2 python3-dev libtool libffi-dev
```

**2. Install MAVProxy**
```bash
pip3 install --user MAVProxy
export PATH=$PATH:~/.local/bin
```

**3. Install SITL Workspace**
Follow these two official guides in order:
* [ArduPilot ROS 2 Guide](https://ardupilot.org/dev/docs/ros2.html)
* [ArduPilot ROS 2 SITL with Gazebo](https://ardupilot.org/dev/docs/ros2-sitl.html)

*Note: Choose ArduPilot 4.5 branch when prompted during the setup.*

**4. Source the Workspace (`.bashrc`)**
To avoid sourcing files manualy in every new terminal, add the following lines to the end of your `~/.bashrc` file:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
``` 
*(Change `ros2_ws` if you named your workspace differently.)*

>**Troubleshooting:** If the `colcon build` step freezes due to RAM limitations (16 GB is sometimes insufficient for parallel building), consider increasing your system swap space to 8GB.

### 2.3 Gazebo Harmonic Installation
Install Gazebo Harmonic following the [ArduPilot Gazebo Installation Guide](https://ardupilot.org/dev/docs/ros2-gazebo.html).

To load custom 3D models into Gazebo, you must add your models directory to the Gazebo resource path. Add this to your `~/.bashrc`:
```bash
export GZ_VERSION=harmonic
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/username/models
``` 
*(Replace `username` with your actual Linux user name)*

You can also add this repository’s model path to GZ_SIM_RESOURCE_PATH, or alternatively copy the models from this repository into your existing models directory.

### 2.4 MAVROS Installation
MAVROS should be built from source for ROS 2 compatibility.
```bash
sudo apt install -y python3-vcstool python3-rosinstall-generator python3-osrf-pycommon

cd ~/ros2_ws
mkdir -p src

# Fetch MAVLink and MAVROS sources
rosinstall_generator --format repos mavlink | tee /tmp/mavlink.repos
rosinstall_generator --format repos --upstream mavros | tee -a /tmp/mavros.repos

# Import repositories and install dependencies
vcs import src < /tmp/mavlink.repos
vcs import src < /tmp/mavros.repos
rosdep update
rosdep install --from-paths src --ignore-src -y

# Install GeographicLib datasets
sudo ./src/mavros/mavros/scripts/install_geographiclib_datasets.sh

# Build MAVROS
colcon build

source install/setup.bash
```

### 2.5 Mission Planner (Optional GUI)
Mission Planner is a highly comprehensive Ground Control Station (GCS) for ArduPilot. It allows visual waypoint creation, parameter tuning, and flight monitoring.
Because it's a Windows application, it runs on Ubuntu using `mono`:
* [Mission Planner Linux Installation Guide](https://ardupilot.org/planner/docs/mission-planner-installation.html)

---

## 3. Running the Simulation

Testing if the installation was successful requires launching Gazebo, SITL, and MAVROS to verify they all interconnect properly.

### Step 1: Launch Gazebo and SITL
In a new terminal, run:
```bash
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
```
This launches the Gazebo simulation environment containing a runway and an Iris drone model, and also spins up ArduPilot SITL in the background.

### Step 2: Launch MAVROS
In a second terminal, establish the bridge between ROS 2 and ArduPilot:
```bash
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@14555
```
*(If the UDP connection fails, SITL might be exposing a TCP port instead. Try `fcu_url:=tcp://127.0.0.1:5760`)*

### Step 3: Verify Connection
In a third terminal, check if the `mavros` node is actively streaming the drone's status:
```bash
ros2 topic echo /mavros/state
```
If the output says `connected: True`, the setup was successful! You can now send velocity setpoints, read visual data, or override controls using standard ROS 2 commands.

### Step 4: Connecting Mission Planner
If you installed Mission Planner, launch it using:
```bash
mono MissionPlanner.exe
```
Select `AUTO` or the correct TCP/UDP port in the top-right corner, and click "Connect". You will see the drone's position on the map.

# GNSS-Denied UAV Navigation via Visual Localization

This repository provides a visual odometry and navigation system for UAVs in GNSS-denied environments. It uses real-time computer vision feature matching (SIFT, ORB, AKAZE, BRISK, HOG) against a previously mapped visual database to estimate the drone's position, and autonomously backtracks if the visual fix is lost.

---

## Prerequisites & Setup

Before installing this package, you must fully set up the ArduPilot SITL, ROS 2, MAVROS, and Gazebo Harmonic simulation environment. 

Please refer to our detailed setup guide first: 👉 **[Simulation Setup Guide](docs/simulation_setup.md)**

---

## Installation

Once your simulation environment is ready and sourced, clone this repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/yakupgulcan/GNSS-Denied-UAV-Navigation.git gnss_denied_nav
cd ~/ros2_ws
colcon build --packages-select gnss_denied_nav
source install/setup.bash
```

---

## Launch Architecture & Workflows

The package provides automated, single-command launch workflows for both mapping and autonomous navigation:

| Launch File | Purpose | Key Nodes Launched |
| :--- | :--- | :--- |
| **`gnss_available_data_collection.launch.py`** | Records survey camera frames + GPS telemetry to build the visual database. | Gazebo, ArduPilot SITL (`gnss_available.param`), MAVROS, `save_frames`, `follow_local_wp` |
| **`gnss_denied_navigation.launch.py`** | Executes autonomous vision-based flight in a GPS-denied environment. | Gazebo, ArduPilot SITL (`gnss_denied.param`), MAVROS, `nav_gcs`, `optical_flow`, `visual_estimator`, `base_controller`, `navigation_runner` |
| **`simulation.launch.py`** | Base simulation bringup. | Gazebo Harmonic server & GUI, ArduPilot SITL (`iris_custom`), MAVROS |

---

## Workflow Guide

### Step 1: Collect Mapping Dataset (GNSS Available)

Before autonomous navigation can operate without GPS, you need a visual database (`.npz`) of the flight area. Run the dedicated data collection launch file:

```bash
ros2 launch gnss_denied_nav gnss_available_data_collection.launch.py
```

This launch file:
1. Launches Gazebo and ArduPilot SITL with GPS enabled (`gnss_available.param`).
2. Connects MAVROS.
3. Automatically starts `save_frames` to record downward-facing camera images synchronized with GPS tags to `~/frames_<timestamp>/`.
4. Executes `follow_local_wp` to autonomously take off, point the camera gimbal straight down (-90°), and fly a parametric survey grid across the flight corridor.

*(Optional)* You can customize the survey area dimensions via launch arguments:
```bash
ros2 launch gnss_denied_nav gnss_available_data_collection.launch.py \
    width_left:=40.0 \
    width_right:=40.0 \
    forward_distance:=600.0 \
    corridor_spacing:=20.0 \
    flight_alt:=30.0
```

---

### Step 2: Build the Visual Feature Database

Process the recorded image frames into a compressed feature database (e.g., using SIFT):

```bash
python3 src/gnss_denied_nav/gnss_denied_nav/build_features/build_feature_database_sift.py
```

Other available extractors in `build_features/`: `build_feature_database_orb.py`, `_akaze.py`, `_brisk.py`, `_hog.py`.

---

### Step 3: Configure Parameters

All tunable parameters are consolidated into a single configuration file at `config/visual_nav_params.yaml`.

Open `config/visual_nav_params.yaml` and set `db_path` to your generated `.npz` file:

```yaml
visual_estimator_node:
  ros__parameters:
    db_path: "/home/username/path/to/your/features_db_sift_2000.npz"
    algorithm: "SIFT"  # Options: SIFT, ORB, AKAZE, BRISK, HOG
    db_path: "~/frames_2026-01-17_17-22-56/features_db_sift_2000.npz"
```

You can also tune flight speed, target coordinates (`target_pos_x`, `target_pos_y`), and PID control gains in this file.

---

### Step 4: Run GNSS-Denied Autonomous Navigation

Once your database is ready and configured, start the full GNSS-denied navigation pipeline with a single command:

```bash
ros2 launch gnss_denied_nav gnss_denied_navigation.launch.py
```

This single command automatically:
1. Starts the Gazebo simulation and loads ArduPilot with GPS disabled and EKF failsafes bypassed (`config/arducopter_params/gnss_denied.param`).
2. Starts MAVROS with the proper FCU bridge configuration.
3. Sequentially launches the vision and control nodes after simulation stabilization:
   - **`nav_gcs`**: Ground station GUI displaying real-time tracking, odometry, and feature matches.
   - **`optical_flow`**: Computes visual velocity and altitude-scaled motion vectors.
   - **`visual_estimator`**: Matches incoming camera frames against your visual database to estimate ENU position and manages breadcrumb backtracking.
   - **`base_controller`**: Translates position and velocity commands into MAVLink RC Overrides (`ALT_HOLD` mode).
   - **`navigation_runner`**: Autonomous planner that tracks waypoints toward the target position.

*(Optional)* You can supply a custom parameters file:
```bash
ros2 launch gnss_denied_nav gnss_denied_navigation.launch.py params_file:=/path/to/custom_params.yaml
```

---

## Flight Control Architecture

Standard ArduPilot waypoint navigation (`AUTO` mode) requires a confident GPS lock. In a GNSS-denied environment:
- The system operates in **`ALT_HOLD`** mode.
- The ROS 2 navigation nodes compute positional errors from the visual estimator.
- Control outputs are sent to the flight controller as simulated joystick commands using **MAVLink RC Overrides** on Roll, Pitch, Throttle, and Yaw channels, allowing autonomous flight while ArduPilot maintains attitude and altitude stabilization.

---

## Repository Structure

```text
gnss_denied_nav/
├── config/
│   ├── arducopter_params/
│   │   ├── gnss_available.param   # ArduPilot parameter overrides (GPS enabled)
│   │   └── gnss_denied.param      # ArduPilot parameter overrides (GPS disabled)
│   └── visual_nav_params.yaml     # Unified parameter configuration for all nodes
├── docs/
│   └── simulation_setup.md        # Environment setup and dependencies guide
├── gnss_denied_nav/
│   ├── build_features/            # Offline database generator scripts
│   ├── *_detect_match.py          # Feature extraction & matching implementations
│   ├── base_controller.py         # RC override PID flight controller
│   ├── follow_local_wp.py         # Parametric survey mission generator
│   ├── nav_gcs.py                 # Ground control visualization GUI
│   ├── navigation_runner.py       # High-level waypoint planner & safety logic
│   ├── optical_flow.py            # Optical flow velocity estimation
│   ├── save_frames.py             # Synchronous image & telemetry recorder
│   └── visual_estimator.py        # Core visual localization node
├── launch/
│   ├── gnss_available_data_collection.launch.py  # Automated mapping pipeline
│   ├── gnss_denied_navigation.launch.py          # Autonomous navigation pipeline
│   ├── iris_custom.launch.py                     # Iris quadcopter spawn & SITL
│   └── simulation.launch.py                      # Simulation & MAVROS bringup
├── models/                        # Gazebo drone & world models
├── worlds/                        # Gazebo simulation worlds
├── package.xml
├── setup.py
└── README.md
```

---

## License

This project is licensed under the Apache 2.0 License.
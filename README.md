# GNSS-Denied UAV Navigation via Visual Localization

This repository provides a visual odometry and navigation system for UAVs in GNSS-denied environments. It matches camera frames against a georeferenced map of survey frames (SIFT, ORB, AKAZE or BRISK features), verifies each match with a RANSAC homography and converts it into a metric ENU position, and dead-reckons with optical flow between fixes.

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

### Map meshes

The Gazebo map meshes (`models/**/*.glb`) are too large for git. Download [models.zip](https://drive.google.com/file/d/1qNm7I9Xv2XJ14P4q1CX50A6-nq0itjjh/view?usp=sharing) (~430 MB) and extract it into the repository's `models/` directory, so that each mesh ends up as `models/<model>/meshes/<model>.glb` next to its `model.sdf`.

Then add the models to the Gazebo resource path:
```bash
echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/ros2_ws/src/gnss_denied_nav/models' >> ~/.bashrc
```

### Python packages

Installed in the user site. Keep `numpy` below 2 (ROS Humble `cv_bridge` is built against numpy 1.x):
```bash
pip install --user "numpy<2" opencv-python pandas matplotlib pymavlink MAVProxy PySide6
pip install --user --no-deps pyqtgraph
```
`pyqtgraph` is installed with `--no-deps` so that pip does not upgrade numpy. Versions in use: numpy 1.26.4, opencv-python 4.13, PySide6 6.10, pyqtgraph 0.14, MAVProxy 1.8.74.

### Network

Gazebo and ROS 2 pick a network interface when they start. If Wi-Fi drops while the simulation is running, the simulation fails. Keep both on the loopback interface by adding to `~/.bashrc`:
```bash
export GZ_IP=127.0.0.1
export ROS_LOCALHOST_ONLY=1
```

### Gimbal camera rate (optional)

The upstream gimbal camera (`ardupilot_gazebo/models/gimbal_small_3d/model.sdf`) publishes at 10 Hz (`<update_rate>10</update_rate>`). It can be raised (for example to 30 Hz) by editing that value. Localization is limited by its processing time (~110 ms per frame), so results are not very sensitive to it, but optical flow gets more frames. Check the rate with `ros2 topic hz /camera/image`.

---

## Launch Architecture & Workflows

The package provides automated, single-command launch workflows for both mapping and autonomous navigation:

| Launch File | Purpose | Key Nodes Launched |
| :--- | :--- | :--- |
| **`gnss_available_data_collection.launch.py`** | Records survey camera frames + GPS telemetry to build the visual database. | Gazebo, ArduPilot SITL (`gnss_available.param`), MAVROS, `save_frames`, `follow_local_wp` |
| **`gnss_denied_navigation.launch.py`** | Executes autonomous vision-based flight in a GPS-denied environment. | Gazebo, ArduPilot SITL (`gnss_denied.param`), MAVROS, `optical_flow`, `visual_estimator`, `base_controller`, `navigation_runner` |
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
3. Automatically starts `save_frames` to record downward-facing camera images synchronized with GPS tags to `~/frames_<timestamp>/` once above safe recording altitude (`min_record_alt:=25.0`).
4. Executes `follow_local_wp` to autonomously take off, point the camera gimbal straight down (-90°), and fly a parametric corridor swath survey across the planned flight path.
5. **Clean Data Finalization**: When the final survey swath waypoint is reached (e.g. at 40, 3000), `follow_local_wp` automatically stops frame recording via `/dataset/record` so the return flight to origin (0, 0) does not pollute the dataset.
6. **Automated Touchdown**: Once the drone arrives back at origin (0, 0), it automatically triggers `RTL` to land and disarm (`rtl_on_finish:=true`).

*(Optional)* You can customize the corridor survey dimensions via launch arguments:
```bash
ros2 launch gnss_denied_nav gnss_available_data_collection.launch.py \
    target_x:=0.0 \
    target_y:=3000.0 \
    corridor_width:=80.0 \
    lane_spacing:=20.0 \
    flight_alt:=30.0
```
*(For a single-lane survey along a specific corridor, e.g. at X=40, set `target_x:=40.0 corridor_width:=0.0`)*

---

### Step 2: Build the Map Database

Extract features from the recorded survey frames into a map database:

```bash
ros2 run gnss_denied_nav build_map_db ~/frames_<timestamp> --method SIFT
```

This writes `~/frames_<timestamp>/map_sift_n2000.npz` (method + settings in the name, so e.g. `--n-features 500` gives `map_sift_n500.npz` next to it). Other methods: `--method ORB | AKAZE | BRISK`, with `--n-features` (SIFT/ORB) or `--threshold` (AKAZE/BRISK).

Check the map offline before flying. This localizes every survey frame against its neighbours (leave-one-out) and reports position error, heading error and latency:

```bash
ros2 run gnss_denied_nav eval_localization --db ~/frames_<timestamp>/map_sift_n2000.npz
```

Add `--queries <other_frames_dir>` to evaluate frames from a different flight, and `--mode global` to test localization without a position prior.

---

### Step 3: Configure Parameters

All tunable parameters are consolidated into a single configuration file at `config/visual_nav_params.yaml`.

Open `config/visual_nav_params.yaml` and set `db_path` to your generated `.npz` file:

```yaml
visual_estimator_node:
  ros__parameters:
    db_path: "~/frames_<timestamp>/map_sift_n2000.npz"  # the feature method is read from the map
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
   - **`optical_flow`**: Computes visual velocity and altitude-scaled motion vectors.
   - **`visual_estimator`**: Matches incoming camera frames against the map database around the predicted position and publishes the ENU position.
   - **`base_controller`**: Translates position and velocity commands into MAVLink RC Overrides (`ALT_HOLD` mode).
   - **`navigation_runner`**: Autonomous planner that tracks waypoints toward the target position.
   - **`perf_metrics_logger`**: Records visual localization errors against Gazebo ground-truth odometry to `perf_metrics.csv` in the run directory.

*(Optional)* You can customize the test world, parameters file, or logger:
```bash
ros2 launch gnss_denied_nav gnss_denied_navigation.launch.py \
    world:=map_changed.sdf \
    params_file:=/path/to/custom_params.yaml \
    enable_logger:=true
```

---

### Step 4b (experimental): Let ArduPilot Fly — GPS_INPUT + GUIDED

Instead of RC overrides in `ALT_HOLD`, this pipeline feeds the visual estimate to ArduPilot as a MAVLink `GPS_INPUT` stream and lets the autopilot's own EKF and position controller fly the mission in `GUIDED`:

```bash
ros2 launch gnss_denied_nav gnss_denied_guided.launch.py
```

- ArduPilot is loaded with `config/arducopter_params/gnss_denied_fake_gps.param` (`GPS1_TYPE=14`, EKF position from GPS, no GPS velocity, arming checks and EKF failsafe enabled).
- **`visual_estimator`** starts localized at the known take-off point (`init_at_home`), so the pose is valid on the ground and the first database fix is gated like any other.
- **`gps_input_bridge`** converts `/visual_gps` to `/mavros/gps_input/gps_input`. If the estimate goes stale the stream stops and ArduPilot's failsafes apply.
- **`guided_mission`** waits for the EKF position, then switches to `GUIDED`, arms, takes off, flies to the target and lands.
- `base_controller` and `navigation_runner` are not used. Add `gui:=false` to run without the Gazebo GUI.

---

### Ground Control Station (optional)

The ground station is not started by the launch files, so runs that only collect logs stay lightweight. Start it by hand in another terminal while a navigation launch is running:

```bash
ros2 run gnss_denied_nav nav_gcs --ros-args \
    --params-file $(ros2 pkg prefix gnss_denied_nav)/share/gnss_denied_nav/config/visual_nav_params.yaml
```

The Navigation tab compares the visual estimate with Gazebo ground truth (`/odometry`): map frames, true / estimated / EKF paths, the matched map frame and search radius, localization status, errors, speeds, altitude and matcher statistics. It needs PySide6 and `pyqtgraph` (see [Python packages](#python-packages)).

---

### Step 5: Performance Analysis & Benchmarking

Every navigation launch writes into its own run directory, `~/gnss_nav_logs/<date>_<time>_<world>/` (change the parent with `log_dir:=<dir>`):
- **`perf_metrics.csv`**: timestamp, algorithm, match duration (ms), status (`MATCH` / `OUTLIER`), estimated and ground-truth (`/odometry`) coordinates, and Euclidean error (m).
- **`visual_log.csv`**: accepted map fixes with inlier count and estimated uncertainty.
- **`run_meta.json`** and copies of the parameter files used.
- ArduPilot SITL / MAVProxy files (`mav.tlog`, `logs/*.BIN`, `eeprom.bin`, ...). The launch starts SITL inside the run directory, so every run also starts from a fresh SITL eeprom.
- **`bag/`**: a rosbag of the navigation topics with `record_bag:=true` (add `bag_images:=true` to include camera images).

```bash
ros2 launch gnss_denied_nav gnss_denied_guided.launch.py record_bag:=true
```

To plot a run report (route, errors, match rate, latency) for the newest run, or for a given CSV:

```bash
python3 src/gnss_denied_nav/gnss_denied_nav/plot_run_report.py [~/gnss_nav_logs/<run>/perf_metrics.csv]
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
│   ├── localization/              # ROS-free localization core (features, matching, geometry, map database)
│   ├── tools/                     # build_map_db, eval_localization
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
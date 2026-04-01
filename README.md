# GNSS-Denied UAV Navigation via Visual Localization

This repository provides a visual odometry and navigation system for UAVs in GNSS-denied environments. It uses real-time computer vision feature matching (SIFT, ORB, AKAZE, BRISK, HOG) against a previously mapped visual database to estimate the drone's position, and autonomously backtracks if the visual fix is lost.

---

## Prerequisites & Setup

Before installing this package, you must fully set up the ArduPilot SITL, ROS 2, MAVROS, and Gazebo Harmonic simulation environment. 

Please refer to our detailed setup guide first:
👉 **[Simulation Setup Guide](docs/simulation_setup.md)**

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

## Configuration

### 1. Configure ArduPilot Parameters (Mission Planner)
For a true GNSS-denied flight using visual odometry, you must explicitly disable the physical GPS and configure the Extended Kalman Filter (EK3) to ignore it. You also need to configure the gimbal settings.

Connect Mission Planner to your SITL instance, go to the **Config** tab > **Full Parameter List**, and apply the following parameters:

**Enable Gimbal Control via GCS:**
Do this before building the feature database and ensure that the camera faces downward when collecting images.
- `MAV_GCS_SYSID` = `1`
- `MAV_GCS_SYSID_HI` = `255`

**Disable GPS & Failsafes:**
Do this step after building the feature database.
- `GPS1_TYPE` = `0` *(Disables the default GPS)*
- `AHRS_GPS_USE` = `0` *(Disables AHRS GPS usage)*
- `EK3_SRC1_POSXY` = `0` *(Stops EK3 from using GPS for position)*
- `EK3_SRC1_VELXY` = `0` *(Stops EK3 from using GPS for velocity)*
- `EK3_SRC1_VELZ` = `0` 
- `ARMING_CHECK` = `0` *(Disables pre-arm checks that require GPS lock)*
- `FS_DR_ENABLE` = `0` *(Disables dead reckoning failsafe)*
- `FS_EKF_ACTION` = `0` *(Disables EKF failsafe action)*
- `FS_EKF_THRESH` = `0`

Click **Write Parameters** and reboot the flight controller.

### 2. Flight Control Architecture (ALT_HOLD & RC Override)
Because standard ArduPilot waypoint navigation (`AUTO` mode) requires a highly confident GPS lock, this system instead operates in **`ALT_HOLD`** mode. 
Our ROS 2 navigation nodes compute the positional error from the visual estimator and translate it into simulated joystick commands using **MAVLink RC Overrides**. 
By overriding the Roll, Pitch, and Yaw channels, the system autonomously flies the drone along the visual trajectory while the flight controller handles basic stabilization.

### 3. Prepare the Visual Database
The navigation system requires a geographic visual database (`.npz` format). 
Due to their large file size, visual databases are *not* included in this Git repository. You must generate them using a prior flight recording:

1. Fly a mapping pass over your target environment with standard GPS enabled.
2. Record frames and telemetry using:
   ```bash
   ros2 run gnss_denied_nav save_frames
   ```
3. Build the database (e.g., using SIFT):
   ```bash
   python3 src/gnss_denied_nav/gnss_denied_nav/build_features/build_feature_database_sift.py
   ```

### 4. Set the DB Path in ROS 2 Parameters
Open the configuration file located at `config/visual_nav_params.yaml` and update the `db_path` variable to point to your newly generated `.npz` file:

```yaml
visual_estimator:
  ros__parameters:
    db_path: "/home/username/path/to/your/features_db_sift_2000.npz"
    algorithm: "SIFT"  # Options: SIFT, ORB, AKAZE, BRISK, HOG
```

---

## Running the System

**1. Start the Simulation**
```bash
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
```

**2. Start MAVROS**
```bash
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@14555
```

**3. Run the Base Controller**
This node interfaces directly with MAVROS to send altitude and velocity RC overrides.
```bash
ros2 run gnss_denied_nav base_controller --ros-args --params-file src/gnss_denied_nav/config/visual_nav_params.yaml
```

**4. Run the Visual Estimator**
This is the core vision pipeline. It consumes camera images, matches them against the database, and publishes `/visual_gps`.
```bash
ros2 run gnss_denied_nav visual_estimator --ros-args --params-file src/gnss_denied_nav/config/visual_nav_params.yaml
```

**5. Run the Navigation Runner**
This node handles autonomous waypoint tracking, PID control, and the safety backtracking logic.
```bash
ros2 run gnss_denied_nav navigation_runner --ros-args --params-file src/gnss_denied_nav/config/visual_nav_params.yaml
```

***(Optional)* Launch the GCS GUI**
For a comprehensive view of the map and feature matches, run:
```bash
ros2 run gnss_denied_nav nav_gcs
```

---

## Repository Structure
- `gnss_denied_nav/`: Contains the core ROS 2 nodes (`visual_estimator.py`, `navigation_runner.py`, `base_controller.py`).
- `gnss_denied_nav/*_detect_match.py`: Algorithm-specific computer vision matching logic.
- `gnss_denied_nav/build_features/`: Pipeline scripts for generating the `.npz` visual feature databases.
- `config/`: YAML parameter files containing easily tunable variables.
- `docs/`: Supplementary documentation and setup guides.

## License
This project is licensed under the Apache 2.0 License.
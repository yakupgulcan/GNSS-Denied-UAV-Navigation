# GNSS-Denied UAV Navigation

A project focused on visual odometry for UAV navigation in GNSS-denied environments.


## Building the Feature Database

Feature databases are not included in the repository (they are
large binary files generated from your specific flight footage).

To build a database, first collect frames during a GPS-enabled flight,
then run the appropriate script:

```bash
# Example for SIFT (recommended):
ros2 run gnss_denied_nav save_frames

# Then build the database:
python3 gnss_denied_nav/build_features/build_feature_database_sift.py \
    --frames-dir /path/to/your/frames \
    --output features_db_sift.npz
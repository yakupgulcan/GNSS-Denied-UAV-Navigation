from setuptools import find_packages, setup
from glob import glob
package_name = 'gnss_denied_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch and World files automaticly
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        
        # Add config files
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yakupgulcan',
    maintainer_email='yakupgulcan5@gmail.com',
    description='UAV for GNSS-Denied Environments',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Core nodes
            'visual_estimator = gnss_denied_nav.visual_estimator:main',
            'navigation_runner = gnss_denied_nav.navigation_runner:main',
            'base_controller = gnss_denied_nav.base_controller:main',
            'nav_gcs = gnss_denied_nav.nav_gcs:main',

            # Algorithm-specific visual estimators
            'orb_visual_estimator = gnss_denied_nav.orb_visual_estimator:main',
            'sift_visual_estimator = gnss_denied_nav.sift_visual_estimator:main',
            'brisk_visual_estimator = gnss_denied_nav.brisk_visual_estimator:main',
            'hog_visual_estimator = gnss_denied_nav.hog_visual_estimator:main',
            'akaze_visual_estimator = gnss_denied_nav.akaze_visual_estimator:main',

            # Utility nodes
            'save_frames = gnss_denied_nav.save_frames:main',
            'save_logs = gnss_denied_nav.save_logs:main',
            'optical_flow = gnss_denied_nav.optical_flow:main',
            'perf_metrics_logger = gnss_denied_nav.perf_metrics_logger:main',
            'takeoff_stabilize_p_controller = gnss_denied_nav.takeoff_stabilize_p_controller:main',
            'follow_waypoints = gnss_denied_nav.follow_waypoints:main',
            'follow_local_wp = gnss_denied_nav.follow_local_wp:main',
            'visual_backtrackt = gnss_denied_nav.visual_backtrackt:main',
        ],
    },
)

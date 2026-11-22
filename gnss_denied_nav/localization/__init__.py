"""
ROS-free visual localization core.

Pipeline: retrieve candidate map frames -> match the query against each
candidate pair-wise -> verify with a RANSAC homography -> convert the
homography into a metric ENU position, heading and uncertainty.

Nothing in this package imports rclpy, so it can be used from ROS nodes,
offline evaluation tools and tests alike.
"""
from gnss_denied_nav.localization.camera import CameraModel
from gnss_denied_nav.localization.database import MapDatabase
from gnss_denied_nav.localization.features import create_extractor
from gnss_denied_nav.localization.localizer import Localizer, LocalizerConfig
from gnss_denied_nav.localization.types import Features, LocalizationResult

__all__ = [
    'CameraModel',
    'Features',
    'LocalizationResult',
    'Localizer',
    'LocalizerConfig',
    'MapDatabase',
    'create_extractor',
]

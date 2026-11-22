"""Pinhole model of the nadir (straight-down) survey/navigation camera."""
import math
from dataclasses import dataclass


@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    # Horizontal field of view (rad). /camera/image comes from the wide
    # gimbal camera (horizontal_fov 2.0 in gimbal_small_3d); this was
    # confirmed by matching survey frames against their GPS tags.
    hfov: float = 2.0

    @property
    def focal_px(self):
        return (self.width / 2.0) / math.tan(self.hfov / 2.0)

    @property
    def center(self):
        return (self.width / 2.0, self.height / 2.0)

    def gsd(self, altitude_m):
        """Ground sampling distance (m/px) at the given height above ground."""
        return altitude_m / self.focal_px

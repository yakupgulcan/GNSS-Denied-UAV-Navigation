"""FlowOdometry on synthetic nadir image sequences with known motion."""
import math

import pytest
from synthetic import ALT, CAM, make_ground, render

from gnss_denied_nav.optical_flow import FlowOdometry

DT = 0.1


@pytest.fixture(scope='module')
def ground():
    return make_ground(seed=3)


def run(ground, v_east, v_north, heading, yaw_rate=0.0, steps=6):
    flow = FlowOdometry(CAM)
    out = []
    for k in range(steps):
        h = heading + yaw_rate * k * DT
        img = render(ground, v_east * k * DT, v_north * k * DT, h)
        vel = flow.update(img, k * DT, ALT, h)
        if vel is not None:
            out.append(vel)
    assert out
    return out[-1]


@pytest.mark.parametrize('v_east,v_north,heading', [
    (0.0, 3.0, 0.0),      # straight north, nose north
    (3.0, 0.0, 90.0),     # straight east, nose east
    (2.0, 2.0, 30.0),     # diagonal, nose off the track
    (-1.5, 0.5, 200.0),
])
def test_velocity_matches_motion(ground, v_east, v_north, heading):
    ve, vn, _, _ = run(ground, v_east, v_north, heading)
    assert ve == pytest.approx(v_east, abs=0.1)
    assert vn == pytest.approx(v_north, abs=0.1)


def test_yaw_rotation_does_not_create_velocity(ground):
    ve, vn, _, _ = run(ground, 0.0, 0.0, 10.0, yaw_rate=20.0)
    assert math.hypot(ve, vn) < 0.1


def test_body_frame(ground):
    _, _, fwd, right = run(ground, 0.0, 3.0, 90.0)  # moving north, nose east
    assert fwd == pytest.approx(0.0, abs=0.1)
    assert right == pytest.approx(-3.0, abs=0.1)  # north is to the left of an east-facing nose


def test_map_heading_offset_tracks_compass_error():
    """The flow heading follows the map heading, not a biased compass."""
    import json

    import rclpy
    from std_msgs.msg import Float64, String

    from gnss_denied_nav.optical_flow import VisualVelocityNode

    rclpy.init()
    try:
        node = VisualVelocityNode()
        node.heading_callback(Float64(data=352.0))      # EKF heading 8 deg off (true 0)

        def fix(hdg, inliers=200):
            node.diagnostics_cb(String(data=json.dumps({'fix_heading_deg': hdg, 'inliers': inliers})))

        assert node.flow_heading() == pytest.approx(352.0)   # no map fix yet: compass
        fix(0.0)
        assert node.flow_heading() == pytest.approx(0.0)     # first fix sets the offset (wraps)
        fix(1.0, inliers=5)                                  # too few inliers: ignored
        fix(40.0)                                            # implausible jump: ignored
        assert node.flow_heading() == pytest.approx(0.0)
        for _ in range(100):
            fix(2.0)                                         # true heading drifts slowly
        assert node.flow_heading() == pytest.approx(2.0, abs=0.05)
        node.destroy_node()
    finally:
        rclpy.shutdown()

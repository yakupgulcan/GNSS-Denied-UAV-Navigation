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

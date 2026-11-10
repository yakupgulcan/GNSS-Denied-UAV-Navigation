"""
Shared geodetic helpers.

Local ENU <-> latitude/longitude conversions using the same flat-earth
scaling as ArduPilot (Location::offset), so positions computed here agree
with what the flight controller and the SITL/Gazebo bridge compute.
"""
import math

# ArduPilot LOCATION_SCALING_FACTOR is 0.011131884502145034 m per 1e-7 deg.
METERS_PER_DEG = 111318.84502145034

# GPS time: epoch 1980-01-06, currently 18 leap seconds ahead of UTC.
GPS_EPOCH_UNIX = 315964800
GPS_LEAP_SECONDS = 18
SECONDS_PER_WEEK = 604800


def latlon_to_enu(lat, lon, origin_lat, origin_lon):
    """Return (east, north) in meters of lat/lon relative to the origin."""
    north = (lat - origin_lat) * METERS_PER_DEG
    east = (lon - origin_lon) * METERS_PER_DEG * math.cos(math.radians(origin_lat))
    return east, north


def enu_to_latlon(east, north, origin_lat, origin_lon):
    """Return (lat, lon) in degrees of a local ENU offset from the origin."""
    lat = origin_lat + north / METERS_PER_DEG
    lon = origin_lon + east / (METERS_PER_DEG * math.cos(math.radians(origin_lat)))
    return lat, lon


def unix_to_gps_week(unix_seconds):
    """Return (gps_week, milliseconds_into_week) for a UNIX timestamp."""
    gps_seconds = unix_seconds - GPS_EPOCH_UNIX + GPS_LEAP_SECONDS
    week = int(gps_seconds // SECONDS_PER_WEEK)
    week_ms = int((gps_seconds - week * SECONDS_PER_WEEK) * 1000.0)
    return week, week_ms

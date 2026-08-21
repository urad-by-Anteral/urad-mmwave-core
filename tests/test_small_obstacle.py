"""Unit tests for the small obstacle detection TLV parser, on synthetic packets."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.small_obstacle import (
    TLV_OCCUPANCY,
    TLV_POINT_CLOUD,
    TLV_PRESENCE,
    TLV_TARGET_INDEX,
    TLV_TARGET_LIST,
    SmallObstacleFrame,
    format_zone_status,
    parse_frame,
    points_to_xyz,
    read_zones,
)


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _point_cloud_tlv(points) -> bytes:
    units = struct.pack("<5f", 0.01, 0.01, 0.1, 0.25, 0.5)
    body = units + b"".join(struct.pack("<2bh2H", *p) for p in points)
    return _tlv(TLV_POINT_CLOUD, body)


def _target_tlv(tid: int, position, confidence: float) -> bytes:
    body = struct.pack(
        "<I9f16f2f",
        tid,
        *position,
        0.1,
        0.2,
        0.0,  # velocity
        0.0,
        0.0,
        0.0,  # acceleration
        *([0.0] * 16),  # error covariance
        1.0,  # gating gain
        confidence,
    )
    return _tlv(TLV_TARGET_LIST, body)


def _occupancy_tlv(mask: int) -> bytes:
    return _tlv(TLV_OCCUPANCY, struct.pack("<I", mask))


def test_parse_occupancy_bitmask():
    frame = parse_frame(_occupancy_tlv(0b101), timestamp=2.0)
    assert frame.occupancy == 0b101
    assert frame.zone_occupied(0) is True
    assert frame.zone_occupied(1) is False
    assert frame.zone_occupied(2) is True
    assert frame.occupied_zones() == [0, 2]
    assert frame.timestamp == 2.0


def test_occupancy_zero_and_missing():
    frame = parse_frame(_occupancy_tlv(0))
    assert frame.occupancy == 0
    assert frame.occupied_zones() == []
    assert frame.zone_occupied(0) is False

    frame = parse_frame(b"")
    assert frame.occupancy is None
    assert frame.occupied_zones() == []
    assert frame.zone_occupied(0) is False
    assert frame.zone_occupied(-1) is False


def test_parse_frame_with_all_tlvs_combined():
    payload = (
        _point_cloud_tlv([(0, 0, 10, 100, 50), (-40, 5, -10, 200, 80)])
        + _target_tlv(1, (0.5, 1.0, -0.3), 0.99)
        + _tlv(TLV_TARGET_INDEX, bytes([1, 1]))
        + _tlv(TLV_PRESENCE, struct.pack("<I", 1))
        + _occupancy_tlv(0b1)
    )
    frame = parse_frame(payload)
    assert frame.points.shape == (2, 5)
    assert len(frame.obstacles) == 1
    assert frame.obstacles[0].tid == 1
    assert frame.obstacles[0].position == pytest.approx((0.5, 1.0, -0.3))
    assert list(frame.tracking.target_index) == [1, 1]
    assert frame.tracking.presence == 1
    assert frame.occupied_zones() == [0]


def test_point_cloud_scaling_matches_people_tracking():
    # (elevation, azimuth, doppler, range, snr) in raw units
    frame = parse_frame(_point_cloud_tlv([(-30, 10, -50, 400, 60)]))
    range_m, azimuth_deg, elevation_deg, doppler, snr = frame.points[0]
    assert range_m == pytest.approx(400 * 0.25)
    assert azimuth_deg == pytest.approx(10 * 0.01 * 180 / np.pi)
    assert elevation_deg == pytest.approx(-30 * 0.01 * 180 / np.pi)
    assert doppler == pytest.approx(-50 * 0.1)
    assert snr == pytest.approx(60 * 0.5)


def test_unknown_tlv_skipped_and_truncated_tlv_aborts():
    unknown = _tlv(1500, b"\xab" * 10)
    frame = parse_frame(unknown + _occupancy_tlv(0b10))
    assert frame.occupancy == 0b10

    truncated = struct.pack("<2I", TLV_OCCUPANCY, 4)  # declares body, none present
    frame = parse_frame(_target_tlv(2, (1.0, 1.0, 0.0), 0.5) + truncated)
    assert len(frame.obstacles) == 1
    assert frame.occupancy is None

    short_body = _tlv(TLV_OCCUPANCY, b"\x01\x00")  # 2 bytes instead of 4
    frame = parse_frame(short_body)
    assert frame.occupancy is None


def test_trailing_alignment_padding_ends_frame(caplog):
    # Radar Toolbox firmwares pad the packet to 32-byte multiples with 0xBE.
    payload = _occupancy_tlv(1) + b"\xbe" * 20
    with caplog.at_level("WARNING"):
        frame = parse_frame(payload)

    assert frame.occupancy == 1
    assert not caplog.records


def test_points_to_xyz_projection():
    # (range, azimuth°, elevation°, doppler, snr)
    points = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 100.0],  # straight ahead -> y=2
            [2.0, 90.0, 0.0, 0.0, 100.0],  # full right -> x=2
            [2.0, 0.0, -30.0, 0.0, 100.0],  # below horizon -> z=-1
        ]
    )
    x, y, z = points_to_xyz(points)
    assert x == pytest.approx([0.0, 2.0, 0.0], abs=1e-9)
    assert y == pytest.approx([2.0, 0.0, 2.0 * np.cos(np.radians(30))], abs=1e-9)
    assert z == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)

    x_empty, y_empty, z_empty = points_to_xyz(np.zeros((0, 5)))
    assert len(x_empty) == 0 and len(y_empty) == 0 and len(z_empty) == 0


def test_read_zones(tmp_path):
    cfg = tmp_path / "chirp.cfg"
    cfg.write_text(
        "% comment\n"
        "sensorStop\n"
        "boundaryBox -1.8 1.8 0.2 1.5 0.0 2.0\n"
        "occStateMach 2 6 8 3 1 5 0 5\n"
        "zoneDef 0 -0.5 0.5 0.2 1.0 -0.5 1.5\n"
        "zoneDef 1 -0.5 0.5 1.0 2.0 -0.5 1.5\n"
        "sensorStart\n"
    )
    zones = read_zones(cfg)
    assert zones == {
        0: (-0.5, 0.5, 0.2, 1.0, -0.5, 1.5),
        1: (-0.5, 0.5, 1.0, 2.0, -0.5, 1.5),
    }


def test_format_zone_status():
    assert format_zone_status(SmallObstacleFrame(), 2) == "n/a"

    frame = SmallObstacleFrame(occupancy=0b01)
    assert format_zone_status(frame, 2) == "0:OCCUPIED 1:clear"

    # More bits set than configured zones: show them all.
    frame = SmallObstacleFrame(occupancy=0b100)
    assert format_zone_status(frame, 1) == "0:clear 1:clear 2:OCCUPIED"

    # No zones configured, nothing occupied: still show zone 0.
    frame = SmallObstacleFrame(occupancy=0)
    assert format_zone_status(frame, 0) == "0:clear"


def test_height_colors_low_points_stand_out():
    from urad_mmwave.apps.small_obstacle_viewer import height_colors, track_color

    colors = height_colors(np.array([-0.5, 1.5]), zmin=-0.5, zmax=1.5)
    assert colors.shape == (2, 3)
    assert colors.dtype == np.uint8
    low, high = colors
    assert low[0] > low[2]  # near-ground point is warm (red > blue)
    assert high[2] > high[0]  # high point is cool (blue > red)

    # Out-of-range heights clamp to the gradient ends.
    clamped = height_colors(np.array([-10.0, 10.0]), zmin=-0.5, zmax=1.5)
    assert (clamped[0] == low).all() and (clamped[1] == high).all()

    # Degenerate range must not divide by zero.
    flat = height_colors(np.array([0.0, 0.0]), zmin=0.0, zmax=0.0)
    assert flat.shape == (2, 3)

    assert track_color(3) == track_color(3)
    assert track_color(0) != track_color(1)

"""Unit tests for the area scanner TLV parser, built on synthetic packets."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.area_scanner import (
    TLV_DYNAMIC_POINTS,
    TLV_DYNAMIC_SIDE_INFO,
    TLV_STATIC_POINTS,
    TLV_STATIC_SIDE_INFO,
    TLV_TARGET_INDEX,
    TLV_TRACK_LIST,
    TRACK_INDEX_NOISE,
    TRACK_INDEX_OUT_OF_BOUNDS,
    AreaTarget,
    ZoneConfig,
    classify_zones,
    parse_frame,
    projected_position,
    zone_of,
)


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _dynamic_tlv(points) -> bytes:
    """Points as (range m, azimuth rad, elevation rad, doppler m/s)."""
    return _tlv(TLV_DYNAMIC_POINTS, b"".join(struct.pack("<4f", *p) for p in points))


def _static_tlv(points) -> bytes:
    """Points as (x m, y m, z m, doppler m/s)."""
    return _tlv(TLV_STATIC_POINTS, b"".join(struct.pack("<4f", *p) for p in points))


def _side_info_tlv(tlv_type: int, infos) -> bytes:
    """Side info as (snr, noise) pairs."""
    return _tlv(tlv_type, b"".join(struct.pack("<2H", *i) for i in infos))


def _target_tlv(*targets) -> bytes:
    """Targets as (tid, posX, posY, velX, velY, accX, accY, posZ, velZ, accZ)."""
    return _tlv(TLV_TRACK_LIST, b"".join(struct.pack("<I9f", *t) for t in targets))


def test_parse_dynamic_points_with_side_info():
    payload = _dynamic_tlv([(4.0, np.pi / 2, -np.pi / 6, -1.5)]) + _side_info_tlv(
        TLV_DYNAMIC_SIDE_INFO, [(120, 30)]
    )
    frame = parse_frame(payload, timestamp=1.0)

    assert frame.dynamic_points.shape == (1, 6)
    range_m, azimuth_deg, elevation_deg, doppler, snr, noise = frame.dynamic_points[0]
    assert range_m == pytest.approx(4.0)
    assert azimuth_deg == pytest.approx(90.0)  # radians converted to degrees
    assert elevation_deg == pytest.approx(-30.0)
    assert doppler == pytest.approx(-1.5)
    assert snr == pytest.approx(120)
    assert noise == pytest.approx(30)
    assert frame.timestamp == 1.0


def test_parse_dynamic_points_without_side_info_pads_zeros():
    frame = parse_frame(_dynamic_tlv([(2.0, 0.0, 0.0, 0.5)]))
    assert frame.dynamic_points.shape == (1, 6)
    assert frame.dynamic_points[0, 4:6] == pytest.approx([0.0, 0.0])


def test_parse_static_points_with_side_info():
    payload = _static_tlv([(1.0, 2.0, -0.5, 0.0)]) + _side_info_tlv(
        TLV_STATIC_SIDE_INFO, [(300, 45)]
    )
    frame = parse_frame(payload)

    assert frame.static_points.shape == (1, 6)
    assert frame.static_points[0] == pytest.approx([1.0, 2.0, -0.5, 0.0, 300, 45])
    assert frame.dynamic_points.shape == (0, 6)


def test_parse_targets_reassembles_xyz_order():
    # The firmware interleaves X/Y first and appends the Z components:
    # tid, posX, posY, velX, velY, accX, accY, posZ, velZ, accZ.
    frame = parse_frame(
        _target_tlv((5, 1.0, 2.0, 0.1, 0.2, 0.01, 0.02, 0.5, 0.3, 0.03))
    )
    assert len(frame.targets) == 1
    target = frame.targets[0]
    assert target.tid == 5
    assert target.position == pytest.approx((1.0, 2.0, 0.5))
    assert target.velocity == pytest.approx((0.1, 0.2, 0.3))
    assert target.acceleration == pytest.approx((0.01, 0.02, 0.03))


def test_parse_target_index_codes():
    body = bytes([0, 3, TRACK_INDEX_OUT_OF_BOUNDS, TRACK_INDEX_NOISE])
    frame = parse_frame(_tlv(TLV_TARGET_INDEX, body))
    assert list(frame.target_index) == [0, 3, 254, 255]


def test_full_frame_with_num_tlvs_and_zero_padding(caplog):
    payload = (
        _dynamic_tlv([(3.0, 0.0, 0.0, 1.0), (5.0, 0.1, 0.0, -1.0)])
        + _side_info_tlv(TLV_DYNAMIC_SIDE_INFO, [(100, 10), (200, 20)])
        + _static_tlv([(0.0, 1.5, 0.0, 0.0)])
        + _side_info_tlv(TLV_STATIC_SIDE_INFO, [(400, 40)])
        + _target_tlv((1, 0.5, 3.0, 0.0, -0.4, 0.0, 0.0, 1.0, 0.0, 0.0))
        + _tlv(TLV_TARGET_INDEX, bytes([1, 1]))
    )
    # Pad to a 32-byte multiple with zero bytes, as the firmware does.
    padded = payload + b"\x00" * (-len(payload) % 32)

    with caplog.at_level("WARNING"):
        frame = parse_frame(padded, num_tlvs=6)
    assert not caplog.records

    assert frame.dynamic_points.shape == (2, 6)
    assert frame.dynamic_points[1, 4:6] == pytest.approx([200, 20])
    assert frame.static_points.shape == (1, 6)
    assert len(frame.targets) == 1
    assert list(frame.target_index) == [1, 1]

    # Without the TLV count the parser stops at the zero padding instead.
    with caplog.at_level("WARNING"):
        frame = parse_frame(padded)
    assert not caplog.records
    assert frame.dynamic_points.shape == (2, 6)
    assert len(frame.targets) == 1


def test_empty_payload_yields_empty_frame():
    frame = parse_frame(b"")
    assert frame.dynamic_points.shape == (0, 6)
    assert frame.static_points.shape == (0, 6)
    assert frame.targets == []
    assert len(frame.target_index) == 0


def test_unknown_tlv_skipped_and_truncated_tlv_aborts():
    unknown = _tlv(6, b"\xab" * 24)  # stats TLV, not decoded by this client
    frame = parse_frame(unknown + _static_tlv([(1.0, 1.0, 0.0, 0.0)]))
    assert frame.static_points.shape == (1, 6)

    truncated = struct.pack("<2I", TLV_TRACK_LIST, 40)  # declares body, none present
    frame = parse_frame(truncated)
    assert frame.targets == []


def test_side_info_longer_than_points_is_clipped():
    payload = _dynamic_tlv([(2.0, 0.0, 0.0, 0.0)]) + _side_info_tlv(
        TLV_DYNAMIC_SIDE_INFO, [(100, 10), (200, 20)]
    )
    frame = parse_frame(payload)
    assert frame.dynamic_points.shape == (1, 6)
    assert frame.dynamic_points[0, 4:6] == pytest.approx([100, 10])


def test_zone_of_boundaries():
    zones = ZoneConfig(critical=(0.0, 2.0), warning=(2.0, 4.0), projection_time=2.0)
    assert zone_of(0.0, zones) == "critical"
    assert zone_of(2.0, zones) == "critical"  # critical wins on the shared edge
    assert zone_of(3.0, zones) == "warning"
    assert zone_of(4.5, zones) == "clear"


def test_projected_position_constant_velocity():
    target = AreaTarget(
        tid=0,
        position=(1.0, 2.0, 0.0),
        velocity=(0.5, -1.0, 0.25),
        acceleration=(0.0, 0.0, 0.0),
    )
    assert projected_position(target, 2.0) == pytest.approx((2.0, 0.0, 0.5))


def test_classify_zones_critical_from_points():
    zones = ZoneConfig()

    dynamic = parse_frame(_dynamic_tlv([(1.5, 0.0, 0.0, 1.0)]))
    assert classify_zones(dynamic, zones).critical

    static = parse_frame(_static_tlv([(0.0, 1.0, 0.5, 0.0)]))  # radial ~1.12 m
    assert classify_zones(static, zones).critical

    # A static point outside the critical zone never triggers any zone.
    far_static = parse_frame(_static_tlv([(0.0, 3.0, 0.0, 0.0)]))
    status = classify_zones(far_static, zones)
    assert not status.critical and not status.warning
    assert status.label == "clear"


def test_classify_zones_warning_from_track_projection():
    zones = ZoneConfig(critical=(0.0, 2.0), warning=(2.0, 4.0), projection_time=2.0)

    # Track at 6 m moving inwards at -1.5 m/s: projected to 3 m -> warning.
    approaching = parse_frame(
        _target_tlv((0, 0.0, 6.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    )
    status = classify_zones(approaching, zones)
    assert status.warning and not status.critical
    assert status.label == "warning"

    # The same track standing still stays clear.
    still = parse_frame(_target_tlv((0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    assert classify_zones(still, zones).label == "clear"

    # A track currently inside the critical zone triggers it.
    inside = parse_frame(_target_tlv((0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    status = classify_zones(inside, zones)
    assert status.critical
    assert status.label == "CRITICAL"


def test_dynamic_points_project_to_xy():
    from urad_mmwave.apps.area_scanner import points_to_xy

    frame = parse_frame(_dynamic_tlv([(2.0, 0.0, 0.0, 0.0)]))
    x, y = points_to_xy(frame.dynamic_points)
    assert x[0] == pytest.approx(0.0, abs=1e-9)
    assert y[0] == pytest.approx(2.0)


def test_read_boundary_boxes_from_area_scanner_config(tmp_path):
    from urad_mmwave.apps.area_scanner import read_boundary_boxes

    cfg = tmp_path / "chirp.cfg"
    cfg.write_text(
        "% Area Scanner configuration\n"
        "sensorStop\n"
        "staticBoundaryBox -8 8 0 8 -1 2\n"
        "boundaryBox -8 8 0 8 -1 2\n"
        "sensorStart\n"
    )
    boxes = read_boundary_boxes(cfg)
    assert boxes["boundaryBox"] == (-8.0, 8.0, 0.0, 8.0, -1.0, 2.0)
    assert boxes["staticBoundaryBox"] == (-8.0, 8.0, 0.0, 8.0, -1.0, 2.0)

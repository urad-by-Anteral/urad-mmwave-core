"""Unit tests for the people tracking TLV parser, built on synthetic packets."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.people_tracking import (
    TLV_POINT_CLOUD,
    TLV_PRESENCE,
    TLV_TARGET_HEIGHT,
    TLV_TARGET_INDEX,
    TLV_TARGET_LIST,
    parse_frame,
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


def test_parse_point_cloud_scaling_and_signed_doppler():
    # (elevation, azimuth, doppler, range, snr) in raw units
    frame = parse_frame(_point_cloud_tlv([(10, -20, -100, 400, 60)]), timestamp=1.0)

    assert frame.points.shape == (1, 5)
    range_m, azimuth_deg, elevation_deg, doppler, snr = frame.points[0]
    assert range_m == pytest.approx(400 * 0.25)
    assert azimuth_deg == pytest.approx(-20 * 0.01 * 180 / np.pi)
    assert elevation_deg == pytest.approx(10 * 0.01 * 180 / np.pi)
    assert doppler == pytest.approx(-100 * 0.1)  # signed, unlike legacy scripts
    assert snr == pytest.approx(60 * 0.5)
    assert frame.timestamp == 1.0


def test_parse_targets():
    frame = parse_frame(_target_tlv(7, (1.0, 2.0, 0.5), 0.9))
    assert len(frame.targets) == 1
    target = frame.targets[0]
    assert target.tid == 7
    assert target.position == pytest.approx((1.0, 2.0, 0.5))
    assert target.velocity == pytest.approx((0.1, 0.2, 0.0))
    assert target.confidence == pytest.approx(0.9)


def test_parse_target_index_and_height_and_presence():
    index = _tlv(TLV_TARGET_INDEX, bytes([0, 1, 1, 255]))
    height = _tlv(TLV_TARGET_HEIGHT, struct.pack("<B3x2f", 3, 1.85, 0.12))
    presence = _tlv(TLV_PRESENCE, struct.pack("<I", 1))
    frame = parse_frame(index + height + presence)

    assert list(frame.target_index) == [0, 1, 1, 255]
    assert frame.heights.shape == (1, 3)
    assert frame.heights[0] == pytest.approx([3, 1.85, 0.12])
    assert frame.presence == 1


def test_parse_frame_with_all_tlvs_combined():
    payload = (
        _point_cloud_tlv([(0, 0, 10, 100, 50), (5, 5, -10, 200, 80)])
        + _target_tlv(1, (0.5, 3.0, 1.0), 0.99)
        + _tlv(TLV_TARGET_INDEX, bytes([1, 1]))
    )
    frame = parse_frame(payload)
    assert frame.points.shape == (2, 5)
    assert len(frame.targets) == 1
    assert list(frame.target_index) == [1, 1]
    assert frame.presence is None


def test_unknown_tlv_skipped_and_truncated_tlv_aborts():
    unknown = _tlv(1500, b"\xab" * 10)
    frame = parse_frame(unknown + _target_tlv(2, (1.0, 1.0, 1.0), 0.5))
    assert len(frame.targets) == 1

    truncated = struct.pack("<2I", TLV_TARGET_LIST, 112)  # declares body, none present
    frame = parse_frame(truncated)
    assert frame.targets == []

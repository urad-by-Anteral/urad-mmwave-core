"""Unit tests for the Automated Parking client, on synthetic packets."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.automated_parking import (
    SUBFRAME_LONG_RANGE,
    SUBFRAME_SHORT_RANGE,
    format_frame,
    nearest_obstacle,
    parking_frames,
    parse_range_profile,
    range_bin_sizes,
)
from urad_mmwave.apps.automated_parking_viewer import height_colors
from urad_mmwave.config import DEFAULT_SYNC_PATTERN, PacketConfig


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _packet_fields(frame_number, num_points, num_tlvs, subframe, payload_len):
    # version, totalPacketLen, platform, frameNumber, time, numObj, numTLVs, subframe
    return (
        0x03060000,
        40 + payload_len,
        0x1843,
        frame_number,
        0,
        num_points,
        num_tlvs,
        subframe,
    )


def _payload(points, side, profile_q9=None):
    body = _tlv(1, b"".join(struct.pack("<4f", *p) for p in points))
    body += _tlv(7, b"".join(struct.pack("<2H", *s) for s in side))
    n = 2
    if profile_q9 is not None:
        body += _tlv(2, np.asarray(profile_q9, dtype="<u2").tobytes())
        n += 1
    body += _tlv(6, bytes(24))  # statistics: skipped
    return body, n + 1


# The two subframes of TI's AWR1843AOP parking configuration
AOP_CFG = [
    "profileCfg 0 77 7 7 44 0 0 15 1 128 6200 0 0 40",
    "profileCfg 1 77 7 3 39 0 0 50 1 128 3600 0 0 30",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "chirpCfg 2 2 1 0 0 0 0 1",
    "chirpCfg 3 3 1 0 0 0 0 2",
    "chirpCfg 4 4 1 0 0 0 0 4",
    "advFrameCfg 2 0 0 1 0",
    "subFrameCfg 0 0 0 2 128 50 0 1 1 50",
    "subFrameCfg 1 0 2 3 128 50 0 1 1 50",
]


def test_range_bin_sizes_follow_subframe_chirp_profile():
    bins = range_bin_sizes(AOP_CFG)
    # 128-point FFT: subframe 0 covers ~62 m, subframe 1 ~10.8 m
    assert bins[0] * 128 == pytest.approx(62.0, rel=0.01)
    assert bins[1] * 128 == pytest.approx(10.79, rel=0.01)
    assert range_bin_sizes(["flushCfg"]) == {}


def test_parse_range_profile_q9_to_db():
    body, n = _payload([], [], profile_q9=[0, 512, 1024])
    db = parse_range_profile(body, n)
    assert db == pytest.approx([0.0, 6.0206, 12.0412], abs=1e-3)
    body, n = _payload([], [])
    assert len(parse_range_profile(body, n)) == 0


def test_nearest_obstacle():
    pts = np.array([[3.0, 4.0, 0, 0, 0, 0], [0.6, 0.8, 1.5, 0, 0, 0]])
    assert nearest_obstacle(pts) == pytest.approx(1.0)  # z is ignored
    assert nearest_obstacle(np.zeros((0, 6))) is None


def test_parking_frames_splits_subframes():
    far, n_far = _payload([(1.0, 30.0, 0.0, -5.0)], [(150, 20)], [100] * 4)
    near, n_near = _payload(
        [(0.5, 2.0, 0.3, 0.0), (-1.0, 4.0, -0.2, 0.0)], [(200, 30), (180, 30)]
    )
    packets = [
        (_packet_fields(7, 1, n_far, SUBFRAME_LONG_RANGE, len(far)), far, 1.0),
        (_packet_fields(7, 2, n_near, SUBFRAME_SHORT_RANGE, len(near)), near, 1.05),
    ]
    frames = list(
        parking_frames(iter(packets), PacketConfig(sync_pattern=DEFAULT_SYNC_PATTERN))
    )
    a, b = frames
    assert a.subframe == SUBFRAME_LONG_RANGE and a.name == "2D 50 m"
    assert a.points[0, :4] == pytest.approx([1.0, 30.0, 0.0, -5.0])
    assert a.points[0, 4:] == pytest.approx([150, 20])
    assert len(a.range_profile_db) == 4
    assert b.subframe == SUBFRAME_SHORT_RANGE and b.points.shape == (2, 6)
    assert b.points[0, 2] == pytest.approx(0.3)  # 3D subframe keeps the height
    assert "nearest obstacle 2.06 m" in format_frame(b)
    assert "nearest" not in format_frame(a)


def test_height_colors_low_blue_high_red():
    c = height_colors([-5.0, 0.5, 9.0])
    assert c[0, 2] == 255 and c[0, 0] == 0  # below the scale: blue
    assert c[1, 1] == 255  # middle: green
    assert c[2, 0] == 255 and c[2, 2] == 0  # above the scale: red

"""Unit tests for the Medium Range Radar TLV parser, built on synthetic packets."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.medium_range_radar import (
    SUBFRAME_MRR,
    SUBFRAME_USRR,
    TLV_CLUSTERS,
    TLV_DETECTED_POINTS,
    TLV_PARKING_ASSIST,
    TLV_TRACKED_OBJECTS,
    parking_assist_to_xy,
    parse_frame,
)

_Q = 7  # xyzQFormat hardcoded by the firmware
_ONE = 1 << _Q


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _points_tlv(points, q_format: int = _Q, declared: int | None = None) -> bytes:
    """points: iterable of (speed_mps, peak_val, x_m, y_m, z_m)."""
    one = 1 << q_format
    body = struct.pack("<2H", declared if declared is not None else len(points), q_format)
    for speed, peak_val, x, y, z in points:
        body += struct.pack(
            "<hH3h",
            round(speed * one),
            peak_val,
            round(x * one),
            round(y * one),
            round(z * one),
        )
    return _tlv(TLV_DETECTED_POINTS, body)


def _clusters_tlv(clusters) -> bytes:
    """clusters: iterable of (x_center, y_center, x_size, y_size) in meters."""
    body = struct.pack("<2H", len(clusters), _Q)
    for values in clusters:
        body += struct.pack("<4h", *(round(v * _ONE) for v in values))
    return _tlv(TLV_CLUSTERS, body)


def _trackers_tlv(trackers) -> bytes:
    """trackers: iterable of (x, y, xd, yd, x_size, y_size) in meters (m/s)."""
    body = struct.pack("<2H", len(trackers), _Q)
    for values in trackers:
        body += struct.pack("<6h", *(round(v * _ONE) for v in values))
    return _tlv(TLV_TRACKED_OBJECTS, body)


def _parking_tlv(ranges_m) -> bytes:
    body = struct.pack("<2H", len(ranges_m), _Q)
    body += b"".join(struct.pack("<H", round(r * _ONE)) for r in ranges_m)
    return _tlv(TLV_PARKING_ASSIST, body)


def test_parse_detected_points_q_scaling_and_signs():
    payload = _points_tlv(
        [
            (-4.5, 900, -2.25, 30.0, 0.5),  # approaching, left of boresight
            (12.0, 100, 10.0, 100.0, -1.0),
        ]
    )
    frame = parse_frame(payload, subframe=SUBFRAME_MRR, timestamp=2.0)

    assert frame.subframe == SUBFRAME_MRR
    assert frame.subframe_name == "MRR"
    assert frame.timestamp == 2.0
    assert frame.points.shape == (2, 5)
    x, y, z, doppler, peak_val = frame.points[0]
    assert x == pytest.approx(-2.25)
    assert y == pytest.approx(30.0)
    assert z == pytest.approx(0.5)
    assert doppler == pytest.approx(-4.5)
    assert peak_val == 900
    assert frame.points[1] == pytest.approx([10.0, 100.0, -1.0, 12.0, 100])


def test_parse_clusters():
    frame = parse_frame(
        _clusters_tlv([(1.5, 10.0, 0.5, 1.0), (-3.0, 5.0, 0.25, 0.25)]),
        subframe=SUBFRAME_USRR,
    )
    assert frame.subframe_name == "USRR"
    assert frame.clusters.shape == (2, 4)
    assert frame.clusters[0] == pytest.approx([1.5, 10.0, 0.5, 1.0])
    assert frame.clusters[1] == pytest.approx([-3.0, 5.0, 0.25, 0.25])


def test_parse_trackers():
    frame = parse_frame(
        _trackers_tlv([(2.0, 40.0, -1.0, -15.0, 0.5, 2.0)]), subframe=SUBFRAME_MRR
    )
    assert len(frame.trackers) == 1
    tracker = frame.trackers[0]
    assert tracker.position == pytest.approx((2.0, 40.0))
    assert tracker.velocity == pytest.approx((-1.0, -15.0))
    assert tracker.size == pytest.approx((0.5, 2.0))


def test_parse_parking_assist_and_projection():
    # 4 bins: boresight, right, far left, left (sin az = 0, 0.5, -1, -0.5)
    frame = parse_frame(_parking_tlv([20.0, 10.0, 20.0, 4.0]), subframe=SUBFRAME_USRR)
    assert frame.parking_assist == pytest.approx([20.0, 10.0, 20.0, 4.0])

    x, y = parking_assist_to_xy(frame.parking_assist)
    # Ordered left to right: sin az = -1, -0.5, 0, 0.5.
    assert x == pytest.approx([-20.0, -2.0, 0.0, 5.0])
    assert y == pytest.approx(
        [0.0, 4.0 * np.sqrt(0.75), 20.0, 10.0 * np.sqrt(0.75)]
    )

    x_empty, y_empty = parking_assist_to_xy(np.zeros(0))
    assert len(x_empty) == 0 and len(y_empty) == 0


def test_parse_combined_usrr_frame():
    payload = (
        _points_tlv([(0.0, 50, 1.0, 2.0, 0.0)])
        + _clusters_tlv([(1.0, 2.0, 0.5, 0.5)])
        + _parking_tlv([20.0] * 32)
    )
    frame = parse_frame(payload, subframe=SUBFRAME_USRR)
    assert frame.points.shape == (1, 5)
    assert frame.clusters.shape == (1, 4)
    assert len(frame.parking_assist) == 32
    assert frame.trackers == []


def test_empty_payload_yields_empty_frame():
    frame = parse_frame(b"", subframe=SUBFRAME_MRR)
    assert frame.points.shape == (0, 5)
    assert frame.clusters.shape == (0, 4)
    assert frame.trackers == []
    assert len(frame.parking_assist) == 0


def test_declared_count_capped_by_body_length():
    # Descriptor declares 5 points but the body carries only 1.
    payload = _points_tlv([(1.0, 10, 0.0, 5.0, 0.0)], declared=5)
    frame = parse_frame(payload)
    assert frame.points.shape == (1, 5)


def test_unknown_tlv_skipped_and_truncated_tlv_aborts():
    unknown = _tlv(9, b"\xab" * 12)
    frame = parse_frame(unknown + _trackers_tlv([(1.0, 1.0, 0.0, 0.0, 0.5, 0.5)]))
    assert len(frame.trackers) == 1

    truncated = struct.pack("<2I", TLV_TRACKED_OBJECTS, 64)  # declares body, none
    frame = parse_frame(truncated)
    assert frame.trackers == []


def test_implausible_tlv_discards_rest_of_frame(caplog):
    garbage = struct.pack("<2I", 12345, 8) + b"\x00" * 8
    with caplog.at_level("WARNING"):
        frame = parse_frame(garbage + _clusters_tlv([(1.0, 1.0, 0.5, 0.5)]))
    assert frame.clusters.shape == (0, 4)
    assert any("Implausible TLV" in r.message for r in caplog.records)


def test_trailing_alignment_padding_ends_frame(caplog):
    # The MRR firmware pads the packet to 32-byte multiples with 0x0F.
    payload = _trackers_tlv([(1.0, 1.0, 0.0, 0.0, 0.5, 0.5)]) + b"\x0f" * 12
    with caplog.at_level("WARNING"):
        frame = parse_frame(payload)

    assert len(frame.trackers) == 1
    assert not caplog.records


def test_implausible_qformat_falls_back_to_default():
    # q = 0 is not a valid firmware value; parser assumes Q7.
    payload = _points_tlv([(0.0, 10, 1.0, 1.0, 0.0)], q_format=0)
    # With q_format=0 the encoder wrote raw integers, so the parser dividing
    # by 2^7 must return value/128.
    frame = parse_frame(payload)
    assert frame.points[0, 0] == pytest.approx(1.0 / 128)


def test_viewer_geometry_helpers():
    from urad_mmwave.apps.medium_range_radar_viewer import (
        _rectangles_xy,
        _segments_xy,
    )

    x, y = _rectangles_xy([(0.0, 0.0, 1.0, 2.0)])
    assert len(x) == 6 and np.isnan(x[-1])
    assert x[:5] == pytest.approx([-1.0, 1.0, 1.0, -1.0, -1.0])
    assert y[:5] == pytest.approx([-2.0, -2.0, 2.0, 2.0, -2.0])

    x, y = _segments_xy([(0.0, 0.0, 3.0, 4.0)])
    assert x[:2] == pytest.approx([0.0, 3.0])
    assert y[:2] == pytest.approx([0.0, 4.0])
    assert np.isnan(x[2]) and np.isnan(y[2])

    x_empty, y_empty = _rectangles_xy([])
    assert len(x_empty) == 0 and len(y_empty) == 0

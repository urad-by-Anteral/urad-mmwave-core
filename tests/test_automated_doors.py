"""Unit tests for the automated doors TLV parser and door state logic."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.automated_doors import (
    HEADER_FORMAT,
    TLV_DYNAMIC_POINTS,
    TLV_DYNAMIC_SIDE_INFO,
    TLV_STATIC_POINTS,
    TLV_STATIC_SIDE_INFO,
    TLV_STATS,
    TLV_TARGET_INDEX,
    TLV_TRACK_LIST,
    AutomatedDoorsFrame,
    DoorConfig,
    DoorStateMachine,
    DoorTarget,
    door_activates,
    find_obstruction,
    parse_frame,
)


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _dynamic_tlv(points) -> bytes:
    """points: iterable of (range m, azimuth rad, elevation rad, doppler)."""
    return _tlv(TLV_DYNAMIC_POINTS, b"".join(struct.pack("<4f", *p) for p in points))


def _static_tlv(points) -> bytes:
    """points: iterable of (x, y, z, doppler)."""
    return _tlv(TLV_STATIC_POINTS, b"".join(struct.pack("<4f", *p) for p in points))


def _side_info_tlv(tlv_type: int, info) -> bytes:
    return _tlv(tlv_type, b"".join(struct.pack("<2h", *v) for v in info))


def _target_tlv(*targets) -> bytes:
    """targets: (tid, (px, py, pz), (vx, vy, vz)) tuples."""
    body = b"".join(
        struct.pack(
            "<I9f",
            tid,
            pos[0],
            pos[1],  # posX, posY
            vel[0],
            vel[1],  # velX, velY
            0.01,
            0.02,  # accX, accY
            pos[2],  # posZ
            vel[2],  # velZ
            0.03,  # accZ
        )
        for tid, pos, vel in targets
    )
    return _tlv(TLV_TRACK_LIST, body)


def _approaching(tid: int = 1, x: float = 0.0, y: float = 2.0, vy: float = -1.0):
    return (tid, (x, y, 1.0), (0.0, vy, 0.0))


def test_header_format_is_44_bytes():
    # Out-of-box header plus the trailing numStaticDetectedObj word.
    assert struct.calcsize(HEADER_FORMAT) == 44


def test_parse_dynamic_points_converts_angles_to_degrees():
    frame = parse_frame(
        _dynamic_tlv([(4.0, np.pi / 6, -np.pi / 12, -0.8)])
        + _side_info_tlv(TLV_DYNAMIC_SIDE_INFO, [(250, -30)]),
        timestamp=1.5,
    )

    assert frame.dynamic_points.shape == (1, 6)
    range_m, azimuth, elevation, doppler, snr, noise = frame.dynamic_points[0]
    assert range_m == pytest.approx(4.0)
    assert azimuth == pytest.approx(30.0)
    assert elevation == pytest.approx(-15.0)
    assert doppler == pytest.approx(-0.8)
    assert snr == pytest.approx(250)  # 0.1 dB steps
    assert noise == pytest.approx(-30)  # side info is signed int16
    assert frame.timestamp == 1.5


def test_parse_static_points_cartesian():
    frame = parse_frame(
        _static_tlv([(0.5, 1.0, -0.3, 0.0), (-1.0, 2.0, 0.0, 0.0)])
        + _side_info_tlv(TLV_STATIC_SIDE_INFO, [(120, 40), (90, 35)])
    )

    assert frame.static_points.shape == (2, 6)
    assert frame.static_points[0, :4] == pytest.approx([0.5, 1.0, -0.3, 0.0])
    assert frame.static_points[1, 4:6] == pytest.approx([90, 35])


def test_parse_targets_field_order():
    # GTRACK_3D appends posZ/velZ/accZ after the 2D fields; make sure the
    # (x, y, z) regrouping is right.
    frame = parse_frame(_target_tlv((7, (1.0, 2.0, 0.5), (0.1, -0.9, 0.05))))

    assert len(frame.targets) == 1
    target = frame.targets[0]
    assert target.tid == 7
    assert target.position == pytest.approx((1.0, 2.0, 0.5))
    assert target.velocity == pytest.approx((0.1, -0.9, 0.05))
    assert target.acceleration == pytest.approx((0.01, 0.02, 0.03))


def test_parse_target_index_and_stats():
    stats = _tlv(TLV_STATS, struct.pack("<6I", 1000, 200, 30000, 4000, 51, 12))
    frame = parse_frame(_tlv(TLV_TARGET_INDEX, bytes([0, 1, 255])) + stats)

    assert list(frame.target_index) == [0, 1, 255]
    assert frame.stats is not None
    assert frame.stats.inter_frame_processing_time_us == 1000
    assert frame.stats.inter_frame_cpu_load_pct == 12


def test_empty_frame():
    frame = parse_frame(b"")
    assert frame.dynamic_points.shape == (0, 6)
    assert frame.static_points.shape == (0, 6)
    assert frame.targets == []
    assert frame.door_open is False
    assert frame.obstructed is False


def test_num_tlvs_stops_before_uninitialized_padding():
    # The firmware pads packets with uninitialized memory, which can look
    # like a plausible TLV header; num_tlvs must prevent reading it.
    padding = struct.pack("<2I", TLV_TRACK_LIST, 40) + b"\x00" * 40
    payload = _target_tlv(_approaching()) + padding
    frame = parse_frame(payload, num_tlvs=1)
    assert len(frame.targets) == 1

    # Without num_tlvs the padding would be decoded as a second TLV.
    frame = parse_frame(payload)
    assert len(frame.targets) == 1  # last TLV wins, still one target


def test_truncated_and_garbage_tlv_tolerated():
    truncated = struct.pack("<2I", TLV_TRACK_LIST, 400)  # declares body, none present
    frame = parse_frame(_dynamic_tlv([(1.0, 0.0, 0.0, 0.0)]) + truncated, num_tlvs=2)
    assert frame.dynamic_points.shape == (1, 6)
    assert frame.targets == []

    garbage = struct.pack("<2I", 9999, 4) + b"\xff" * 4
    frame = parse_frame(garbage + _target_tlv(_approaching()), num_tlvs=2)
    assert frame.targets == []  # implausible TLV aborts the rest of the frame

    frame = parse_frame(b"\xbe" * 32, num_tlvs=1)
    assert frame.targets == []  # 0xBE padding word ends the frame quietly


def test_door_activates_only_when_approaching_in_zone():
    door = DoorConfig()

    def target(x, y, vy):
        return DoorTarget(1, (x, y, 1.0), (0.0, vy, 0.0), (0.0, 0.0, 0.0))

    assert door_activates(target(0.0, 2.0, -1.0), door)  # 2 s away, approaching
    assert not door_activates(target(0.0, 2.0, 1.0), door)  # walking away
    assert not door_activates(target(0.0, 2.0, 0.0), door)  # static (no crash)
    assert not door_activates(target(2.5, 2.0, -1.0), door)  # outside |x| bound
    assert not door_activates(target(0.0, 4.0, -1.0), door)  # beyond zone depth
    assert not door_activates(target(0.0, 3.4, -1.0), door)  # 3.4 s away: too early
    assert not door_activates(target(0.0, -0.5, -1.0), door)  # behind the door


def test_door_state_machine_open_and_hold():
    machine = DoorStateMachine(DoorConfig(hold_frames=2))

    frame = machine.update(parse_frame(_target_tlv(_approaching(tid=4))))
    assert frame.door_open is True
    assert frame.activated_tids == (4,)

    # No activations: the door stays open for hold_frames frames, then closes.
    for _ in range(2):
        frame = machine.update(parse_frame(b""))
        assert frame.door_open is True
        assert frame.activated_tids == ()
    frame = machine.update(parse_frame(b""))
    assert frame.door_open is False


def test_door_stays_closed_for_departing_track():
    machine = DoorStateMachine()
    departing = (2, (0.0, 1.0, 1.0), (0.0, 0.8, 0.0))
    frame = machine.update(parse_frame(_target_tlv(departing)))
    assert frame.door_open is False
    assert frame.door_label == "CLOSED"


def test_static_obstruction_by_distance():
    door = DoorConfig()
    near = parse_frame(_static_tlv([(0.3, 1.0, -0.5, 0.0)]))  # 1.20 m away
    far = parse_frame(_static_tlv([(1.0, 2.0, 0.0, 0.0)]))  # 2.24 m away
    assert find_obstruction(near, door) is True
    assert find_obstruction(far, door) is False
    assert find_obstruction(AutomatedDoorsFrame(), door) is False

    machine = DoorStateMachine(door)
    frame = machine.update(near)
    assert frame.obstructed is True
    assert frame.door_open is False
    assert frame.door_label == "OBSTRUCTED"


def test_door_labels():
    frame = AutomatedDoorsFrame()
    assert frame.door_label == "CLOSED"
    frame.door_open = True
    assert frame.door_label == "OPEN"
    frame.obstructed = True
    assert frame.door_label == "OPEN (obstructed)"


def test_combined_frame_end_to_end():
    payload = (
        _dynamic_tlv([(2.0, 0.0, 0.0, -1.0), (2.5, 0.1, 0.0, -1.1)])
        + _side_info_tlv(TLV_DYNAMIC_SIDE_INFO, [(200, 30), (180, 28)])
        + _static_tlv([(0.2, 0.9, -0.4, 0.0)])
        + _side_info_tlv(TLV_STATIC_SIDE_INFO, [(150, 45)])
        + _target_tlv(_approaching(tid=3, y=1.5, vy=-0.6))
        + _tlv(TLV_TARGET_INDEX, bytes([3, 3]))
    )
    frame = DoorStateMachine().update(parse_frame(payload, timestamp=2.0, num_tlvs=6))

    assert frame.dynamic_points.shape == (2, 6)
    assert frame.static_points.shape == (1, 6)
    assert len(frame.targets) == 1
    assert list(frame.target_index) == [3, 3]
    assert frame.door_open is True  # 2.5 s from the door
    assert frame.obstructed is True  # static point at ~1.0 m
    assert frame.door_label == "OPEN (obstructed)"


def test_door_indicator_colors():
    from urad_mmwave.apps.automated_doors_viewer import door_color

    assert door_color(True, False) != door_color(False, False)
    assert door_color(False, True) != door_color(False, False)
    assert door_color(True, True) == door_color(True, False)  # open wins

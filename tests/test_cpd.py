"""Unit tests for the CPD parser, state machine and classifier (synthetic data)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from urad_mmwave.apps.cpd import (
    DECISION_ADULT,
    DECISION_CHILD,
    DECISION_NONE,
    TLV_POINT_CLOUD,
    CabinSetup,
    CabinZone,
    ClassificationParams,
    OccupancyTracker,
    SensorPosition,
    StateMachineParams,
    assign_zones,
    format_zone_summary,
    parse_frame,
    points_to_cabin,
    read_cabin_setup,
)

# Units as hardcoded in the CPD firmware (mss_main.c).
_FIRMWARE_UNITS = (0.01, 0.01, 0.00028, 0.00025, 0.04)


def _tlv(tlv_type: int, body: bytes) -> bytes:
    # The CPD firmware's TLV length INCLUDES the 8-byte TLV header.
    return struct.pack("<2I", tlv_type, len(body) + 8) + body


def _point_cloud_tlv(points, units=_FIRMWARE_UNITS) -> bytes:
    body = struct.pack("<5f", *units)
    body += b"".join(struct.pack("<2bh2H", *p) for p in points)
    return _tlv(TLV_POINT_CLOUD, body)


def test_parse_point_cloud_scaling():
    # Raw fields: (elevation, azimuth, doppler, range, snr)
    frame = parse_frame(
        _point_cloud_tlv([(10, -20, -100, 4000, 250)]), timestamp=1.0, frame_number=7
    )

    assert frame.points.shape == (1, 5)
    range_m, azimuth, elevation, doppler, snr = frame.points[0]
    assert range_m == pytest.approx(4000 * 0.00025)
    assert azimuth == pytest.approx(-20 * 0.01)  # radians
    assert elevation == pytest.approx(10 * 0.01)  # radians
    assert doppler == pytest.approx(-100 * 0.00028)  # signed
    assert snr == pytest.approx(250 * 0.04)
    assert frame.timestamp == 1.0
    assert frame.frame_number == 7


def test_parse_empty_frame_and_no_point_tlv():
    assert parse_frame(b"").points.shape == (0, 5)

    # A frame carrying only an unknown TLV yields no points.
    frame = parse_frame(_tlv(9, b"\x00" * 16))
    assert frame.points.shape == (0, 5)
    assert frame.zones == []


def test_unknown_tlv_skipped_by_inclusive_length():
    payload = _tlv(9, b"\xab" * 12) + _point_cloud_tlv([(0, 0, 0, 1000, 100)])
    frame = parse_frame(payload)
    assert frame.points.shape == (1, 5)


def test_truncated_garbage_and_padding_tolerance(caplog):
    # Declares a bigger body than present: dropped without raising.
    truncated = struct.pack("<2I", TLV_POINT_CLOUD, 500)
    assert parse_frame(truncated).points.shape == (0, 5)

    # Implausible type and sub-header length both abort the frame cleanly.
    assert parse_frame(struct.pack("<2I", 4242, 16)).points.shape == (0, 5)
    assert parse_frame(struct.pack("<2I", TLV_POINT_CLOUD, 4)).points.shape == (0, 5)

    # 0xBE alignment padding after a valid TLV ends the frame silently.
    caplog.clear()
    payload = _point_cloud_tlv([(0, 0, 0, 1000, 100)]) + b"\xbe" * 12
    with caplog.at_level("WARNING"):
        frame = parse_frame(payload)
    assert frame.points.shape == (1, 5)
    assert not caplog.records


def test_read_cabin_setup(tmp_path):
    cfg = tmp_path / "chirp.cfg"
    cfg.write_text(
        "% comment\n"
        "sensorStop\n"
        "numZones 5\n"
        "totNumRows 2\n"
        "sensorPosition 0 1.2 1.1 90 0 0\n"
        "occStateMach  0 10 10 3 50 1 3 12 20 4 700.0\n"
        "occStateMach  1 10 10 3 50 3 3 12 15 3 700.0\n"
        "classParam  1 50 36 36 36 40 36 0.025 0.025 0.025 0.022 0.025\n"
        "interiorBounds -0.9 0.9 0.0 2.2\n"
        "cuboidDef 1 1   0.15 0.75    0.6 1.2   0.85  1.1\n"
        "cuboidDef 1 2   0.2 0.75    0.3 1.1    0.4  0.85\n"
        "cuboidDef 2 1  -0.75 -0.15   0.6 1.2   0.85  1.1\n"
        "cuboidDef 3 1   0.30 0.8    1.6 2.2    0.80  1.1\n"
        "cuboidDef 4 1   0.0 0.0  0.0 0.0  0.0 0.0\n"  # NULL zone
        "cuboidDef 5 1  -0.8 -0.30   1.6 2.2    0.80  1.1\n"
        "zoneNeighDef 4  1 2  3 5\n"
        "zoneNeighDef 5  0 1  4\n"
        "sensorStart\n"
    )
    setup = read_cabin_setup(cfg)

    assert setup.num_zones == 5
    assert setup.tot_num_rows == 2
    assert setup.sensor_position == SensorPosition(0, 1.2, 1.1, 90, 0, 0)
    assert setup.interior_bounds == (-0.9, 0.9, 0.0, 2.2)

    assert len(setup.zones) == 5
    assert setup.zones[0].cuboids == [
        (0.15, 0.75, 0.6, 1.2, 0.85, 1.1),
        (0.2, 0.75, 0.3, 1.1, 0.4, 0.85),
    ]
    assert setup.zones[3].is_null
    assert not setup.zones[0].is_null
    assert setup.zones[3].zone_type == 1
    assert setup.zones[3].neighbors == (3, 5)
    assert setup.zones[4].neighbors == (4,)

    assert set(setup.state_machines) == {0, 1}
    sm = setup.state_machines[0]
    assert (sm.enter_points_1, sm.enter_snr_1) == (10, 10)
    assert (sm.enter_points_2, sm.enter_snr_2) == (3, 50)
    assert sm.entry_frames == 1
    assert (sm.stay_points, sm.stay_snr) == (3, 12)
    assert (sm.forget_frames, sm.forget_points) == (20, 4)
    assert sm.overload_snr == 700.0
    assert setup.state_machines[1].entry_frames == 3

    assert setup.classification.enabled
    assert setup.classification.num_frame_avg == 50
    assert setup.classification.snr_thresholds == (36, 36, 36, 40, 36)
    assert setup.classification.volume_thresholds == (
        0.025,
        0.025,
        0.025,
        0.022,
        0.025,
    )


def test_read_cabin_setup_rejects_non_cpd_config(tmp_path):
    cfg = tmp_path / "chirp.cfg"
    cfg.write_text("sensorStop\nsensorStart\n")
    with pytest.raises(ValueError, match="no occupancy zones"):
        read_cabin_setup(cfg)


def test_read_cabin_setup_skips_malformed_commands(tmp_path, caplog):
    # TI's own vod_6843_aop_overhead_3row_bus.cfg shipped with a trailing-dot
    # typo; a malformed command must be skipped with a warning, not crash.
    cfg = tmp_path / "chirp.cfg"
    cfg.write_text(
        "numZones 2\n"
        "occStateMach  0 10 10 3 50 1 3 12 20 4 700.0\n"
        "cuboidDef 1 1  -0.48. -0.15 0.4 1.0 0.8 1.3\n"  # malformed -> skipped
        "cuboidDef 1 2  -0.48 -0.15 0.2 0.8 0.4 0.95\n"
        "cuboidDef 2 1  0.15 0.48 0.4 1.0 0.8 1.3\n"
    )
    with caplog.at_level("WARNING"):
        setup = read_cabin_setup(cfg)
    assert any("malformed" in record.message for record in caplog.records)
    assert setup.zones[0].cuboids == [(-0.48, -0.15, 0.2, 0.8, 0.4, 0.95)]
    assert len(setup.zones[1].cuboids) == 1


def test_points_to_cabin_overhead_transform():
    # Overhead mounting: sensor at (0, 1.2, 1.1), rotated 90° in the y-z
    # plane to face the floor. A point straight ahead at 1.1 m must land on
    # the floor directly below the sensor.
    sensor = SensorPosition(0, 1.2, 1.1, 90, 0, 0)
    points = np.array([[1.1, 0.0, 0.0, 0.0, 100.0]])  # range, azim, elev, dopp, snr
    cabin = points_to_cabin(points, sensor)
    assert cabin.shape == (1, 4)
    assert cabin[0, :3] == pytest.approx([0.0, 1.2, 0.0], abs=1e-9)
    assert cabin[0, 3] == 100.0

    # Without rotation the spherical-to-cartesian projection is standard.
    flat = points_to_cabin(
        np.array([[2.0, np.pi / 2, 0.0, 0.0, 50.0]]), SensorPosition()
    )
    assert flat[0, :3] == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)

    assert points_to_cabin(np.zeros((0, 5)), sensor).shape == (0, 4)


def test_assign_zones_strict_bounds():
    zones = [
        CabinZone(zone_id=1, cuboids=[(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)]),
        CabinZone(zone_id=2, cuboids=[(2.0, 3.0, 0.0, 1.0, 0.0, 1.0)]),
    ]
    points = np.array(
        [
            [0.5, 0.5, 0.5, 10.0],  # inside zone 1
            [1.0, 0.5, 0.5, 10.0],  # on the boundary -> excluded (strict)
            [2.5, 0.5, 0.5, 10.0],  # inside zone 2
            [5.0, 5.0, 5.0, 10.0],  # nowhere
        ]
    )
    zone_map = assign_zones(points, zones)
    assert zone_map[:, 0].tolist() == [True, False, False, False]
    assert zone_map[:, 1].tolist() == [False, False, True, False]

    assert assign_zones(np.zeros((0, 4)), zones).shape == (0, 2)


def _make_setup(
    entry_frames: int = 1,
    classification: ClassificationParams | None = None,
    neighbors: tuple[int, ...] = (),
) -> CabinSetup:
    """Two side-by-side zones with simple thresholds, no mounting rotation."""
    params = StateMachineParams(
        enter_points_1=5,
        enter_snr_1=10,
        enter_points_2=2,
        enter_snr_2=50,
        entry_frames=entry_frames,
        stay_points=3,
        stay_snr=8,
        forget_frames=2,
        forget_points=2,
        overload_snr=700.0,
    )
    zones = [
        CabinZone(zone_id=1, cuboids=[(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)]),
        CabinZone(
            zone_id=2, cuboids=[(2.0, 3.0, 0.0, 1.0, 0.0, 1.0)], neighbors=neighbors
        ),
    ]
    return CabinSetup(
        num_zones=2,
        zones=zones,
        state_machines={0: params},
        sensor_position=SensorPosition(),
        classification=classification
        or ClassificationParams(False, 1, (0.0, 0.0), (0.0, 0.0)),
    )


def _frame_with_cabin_points(xyz_snr: list[tuple[float, float, float, float]]):
    """Build a CpdFrame whose spherical points project onto the given x/y/z.

    Uses the inverse of the identity mounting: range/azimuth/elevation such
    that the standard projection lands on the requested cartesian point.
    """
    from urad_mmwave.apps.cpd import CpdFrame

    rows = []
    for x, y, z, snr in xyz_snr:
        rng = float(np.sqrt(x * x + y * y + z * z))
        elevation = float(np.arcsin(z / rng)) if rng else 0.0
        azimuth = float(np.arctan2(x, y))
        rows.append([rng, azimuth, elevation, 0.0, snr])
    return CpdFrame(points=np.array(rows) if rows else np.zeros((0, 5)))


def _occupy_zone1(snr: float = 20.0, count: int = 6):
    return _frame_with_cabin_points([(0.5, 0.5, 0.5, snr)] * count)


def test_state_machine_enter_and_leave():
    tracker = OccupancyTracker(_make_setup(entry_frames=2))

    # One qualifying frame is not enough (entry_frames = 2).
    zones = tracker.update(_occupy_zone1())
    assert not zones[0].occupied
    zones = tracker.update(_occupy_zone1())
    assert zones[0].occupied
    assert zones[0].label == "occupied"
    assert zones[0].num_points == 6
    assert zones[0].avg_snr == pytest.approx(20.0)
    assert not zones[1].occupied

    # A non-qualifying frame in between resets the entry counter.
    tracker = OccupancyTracker(_make_setup(entry_frames=2))
    tracker.update(_occupy_zone1())
    tracker.update(_frame_with_cabin_points([]))
    zones = tracker.update(_occupy_zone1())
    assert not zones[0].occupied

    # Leaving: forget_frames misses (below forget_points) are tolerated.
    tracker = OccupancyTracker(_make_setup(entry_frames=1))
    tracker.update(_occupy_zone1())
    empty = _frame_with_cabin_points([])
    for _ in range(3):  # detect_to_free_count reaches forget_frames + 1
        zones = tracker.update(empty)
        assert zones[0].occupied
    zones = tracker.update(empty)
    assert not zones[0].occupied
    assert zones[0].label == "empty"


def test_state_machine_neighbor_snr_condition():
    # Zone 2 declares zone 1 as neighbor: entering via condition 2 (few
    # points, high SNR) requires beating the neighbor's average SNR.
    setup = _make_setup(entry_frames=1, neighbors=(1,))
    tracker = OccupancyTracker(setup)

    # 3 points in zone 2 at SNR 60 (> enter_snr_2 = 50), but zone 1 has a
    # stronger echo (SNR 80) -> zone 2 must NOT enter.
    frame = _frame_with_cabin_points(
        [(2.5, 0.5, 0.5, 60.0)] * 3 + [(0.5, 0.5, 0.5, 80.0)] * 3
    )
    zones = tracker.update(frame)
    assert not zones[1].occupied

    # Same zone 2 cluster with a weak neighbor -> enters.
    tracker = OccupancyTracker(setup)
    frame = _frame_with_cabin_points(
        [(2.5, 0.5, 0.5, 60.0)] * 3 + [(0.5, 0.5, 0.5, 5.0)] * 3
    )
    zones = tracker.update(frame)
    assert zones[1].occupied


def test_overload_freezes_all_zones():
    tracker = OccupancyTracker(_make_setup(entry_frames=1))

    # Average SNR at the overload threshold freezes every zone: despite the
    # qualifying cluster, no zone may change state this frame.
    zones = tracker.update(_occupy_zone1(snr=700.0))
    assert not zones[0].occupied

    # The freeze counter (2 frames) decrements before each update, so the
    # first post-overload frame is still frozen and the second may enter
    # (same timing as the MATLAB visualizer).
    zones = tracker.update(_occupy_zone1(snr=20.0))
    assert not zones[0].occupied
    zones = tracker.update(_occupy_zone1(snr=20.0))
    assert zones[0].occupied


def test_classification_child_and_adult():
    classification = ClassificationParams(
        enabled=True,
        num_frame_avg=2,
        snr_thresholds=(36.0, 36.0),
        volume_thresholds=(0.025, 0.025),
    )
    setup = _make_setup(entry_frames=1, classification=classification)

    # Child: compact cluster (tiny volume), low total SNR.
    # 6 points x SNR 20 -> total 120/frame; 10*log10(120) = 20.8 dB < 36.
    tracker = OccupancyTracker(setup)
    tracker.update(_occupy_zone1(snr=20.0))  # enters occupied; 1st accumulation
    zones = tracker.update(_occupy_zone1(snr=20.0))  # 2nd -> decision
    assert zones[0].decision == DECISION_CHILD
    assert zones[0].label == "child"
    assert zones[0].decision_snr_db == pytest.approx(10 * np.log10(120.0))
    assert zones[0].decision_volume == pytest.approx(0.0)

    # Adult: high total SNR trips the SNR threshold.
    # 6 points x SNR 690 -> total 4140/frame; 36.2 dB > 36 (avg 690 stays
    # below the 700 overload threshold).
    tracker = OccupancyTracker(setup)
    tracker.update(_occupy_zone1(snr=690.0))
    zones = tracker.update(_occupy_zone1(snr=690.0))
    assert zones[0].decision == DECISION_ADULT
    assert zones[0].label == "adult"

    # Adult by volume: spread cluster exceeds the volume threshold.
    spread = _frame_with_cabin_points(
        [(0.1, 0.1, 0.1, 20.0)] * 3 + [(0.9, 0.9, 0.9, 20.0)] * 3
    )
    tracker = OccupancyTracker(setup)
    tracker.update(spread)
    zones = tracker.update(spread)
    assert zones[0].decision == DECISION_ADULT

    # Leaving the occupied state clears the decision and accumulators.
    tracker = OccupancyTracker(setup)
    tracker.update(_occupy_zone1(snr=20.0))
    tracker.update(_occupy_zone1(snr=20.0))
    empty = _frame_with_cabin_points([(5.0, 5.0, 5.0, 1.0)])
    for _ in range(4):
        zones = tracker.update(empty)
    assert not zones[0].occupied
    assert zones[0].decision == DECISION_NONE


def test_format_zone_summary():
    from urad_mmwave.apps.cpd import ZoneStatus

    zones = [
        ZoneStatus(zone_id=1, occupied=True, decision=DECISION_ADULT),
        ZoneStatus(zone_id=2),
        ZoneStatus(zone_id=3, occupied=True, decision=DECISION_CHILD),
    ]
    assert format_zone_summary(zones) == "Z1:adult Z2:empty Z3:child"


def test_viewer_zone_state_colors_stable_and_distinct():
    from urad_mmwave.apps.cpd_viewer import zone_label_text, zone_state_color

    labels = ("empty", "occupied", "adult", "child")
    colors = [zone_state_color(label) for label in labels]
    assert len(set(colors)) == len(labels)
    assert zone_state_color("weird") == zone_state_color("empty")

    from urad_mmwave.apps.cpd import ZoneStatus

    status = ZoneStatus(
        zone_id=2,
        occupied=True,
        num_points=12,
        decision=DECISION_CHILD,
        decision_snr_db=21.5,
        decision_volume=0.012,
    )
    text = zone_label_text(status)
    assert "Z2 CHILD" in text
    assert "12 pts" in text
    assert "21.5" in text
    assert zone_label_text(ZoneStatus(zone_id=1)) == "Z1 EMPTY"

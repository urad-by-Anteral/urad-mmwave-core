"""Unit tests for the Short Range Radar zone logic, chirp adaptation and replay."""

from __future__ import annotations

import numpy as np
import pytest

from urad_mmwave.apps.short_range_radar import (
    NUM_RINGS,
    NUM_SECTORS,
    DetectedZonesWriter,
    GridConfig,
    SectorDistancesWriter,
    ZoneDetector,
    ZoneEventsWriter,
    ZoneMonitor,
    adapt_commands,
    flatten_zones,
    format_distances,
    main,
    read_pointcloud_file,
    replay_frames,
    sector_distances,
    tx_mode_chirp_path,
    zone_events,
    zone_indices,
    zone_label,
)
from urad_mmwave.apps.short_range_radar_viewer import proximity_level, zone_outline


def _pts(*xy):
    """Points from (x, y) pairs, with z=v=0, snr=100, noise=10."""
    return np.array([[x, y, 0.0, 0.0, 100, 10] for x, y in xy], dtype=float)


def _polar(r, az_deg):
    a = np.radians(az_deg)
    return (r * np.sin(a), r * np.cos(a))


# ---------------------------------------------------------------- grid
def test_grid_config_validation():
    with pytest.raises(ValueError):
        GridConfig(max_range=11)
    with pytest.raises(ValueError):
        GridConfig(fov=120)
    with pytest.raises(ValueError):
        GridConfig(min_hits=0)


@pytest.mark.parametrize(
    "fov, sectors", [(30, [2]), (90, [1, 2, 3]), (150, [0, 1, 2, 3, 4])]
)
def test_sectors_inside_fov(fov, sectors):
    assert list(GridConfig(fov=fov).sectors) == sectors


def test_num_rings_counts_partial_ring():
    assert GridConfig(max_range=4).num_rings == 4
    assert GridConfig(max_range=4.5).num_rings == 5


@pytest.mark.parametrize(
    "az, sector",
    [
        (-70, 0),
        (-46, 0),
        (-44, 1),
        (-16, 1),
        (-14, 2),
        (0, 2),
        (14, 2),
        (16, 3),
        (44, 3),
        (46, 4),
        (74, 4),
    ],
)
def test_sector_boundaries_are_30_degrees_wide(az, sector):
    rings, sectors, inside = zone_indices(_pts(_polar(2.5, az)), GridConfig())
    assert inside[0]
    assert sectors[0] == sector
    assert rings[0] == 2


def test_right_edge_is_not_folded_into_left_sector():
    # the legacy script put azimuths below 15 deg from the X axis (the right
    # edge, here +75..+90 from boresight) into the leftmost sector
    rings, sectors, inside = zone_indices(_pts(_polar(3.0, 70)), GridConfig())
    assert sectors[0] == NUM_SECTORS - 1


def test_points_outside_area_are_ignored():
    grid = GridConfig(max_range=4, fov=90)
    pts = _pts(
        _polar(3.0, 0),  # inside
        _polar(4.5, 0),  # beyond max_range
        _polar(2.0, 60),  # outside the +-45 deg FoV
        (0.5, -1.0),  # behind the radar
    )
    _, _, inside = zone_indices(pts, grid)
    assert inside.tolist() == [True, False, False, False]


def test_no_one_meter_offset():
    # the legacy code subtracted 1 m from Y; a point 0.5 m ahead must land in ring 0
    rings, _, inside = zone_indices(_pts((0.0, 0.5)), GridConfig())
    assert inside[0] and rings[0] == 0


# ---------------------------------------------------------------- detector
def test_zone_needs_consecutive_hits():
    det = ZoneDetector(GridConfig(min_hits=4))
    target = _pts(_polar(1.5, 0))
    for _ in range(3):
        assert not det.update(target).any()
    detected = det.update(target)
    assert detected[1, 2] and detected.sum() == 1


def test_one_empty_frame_resets_the_count():
    det = ZoneDetector(GridConfig(min_hits=3))
    target = _pts(_polar(1.5, 0))
    det.update(target)
    det.update(target)
    det.update(np.empty((0, 6)))
    det.update(target)
    assert not det.update(target).any()
    assert det.update(target)[1, 2]


def test_several_points_in_one_zone_count_once():
    det = ZoneDetector(GridConfig(min_hits=2))
    cluster = _pts(_polar(1.2, 1), _polar(1.4, -2), _polar(1.8, 5))
    det.update(cluster)
    assert det.hits[1, 2] == 1


def test_flatten_order_far_to_near():
    detected = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=bool)
    detected[0, 0] = True  # nearest ring, leftmost sector
    detected[9, 4] = True  # farthest ring, rightmost sector
    flat = flatten_zones(detected)
    assert len(flat) == 50
    assert flat[4] == 1  # first row = farthest ring, last column
    assert flat[45] == 1  # last row = nearest ring, first column
    assert sum(flat) == 2


def test_zone_label():
    assert zone_label(2, 2) == "2-3 m, -15..+15 deg"


# ---------------------------------------------------------------- chirp
CFG = [
    "flushCfg",
    "profileCfg 0 77 8 7 200 0 0 20 1 384 2000 0 0 30",
    "aoaFovCfg -1 -90 90 -150 150",
    "cfarFovCfg -1 0 0 10",
    "cfarFovCfg -1 1 -4.68 4.68",
    "sensorStart",
]


def test_adapt_commands_rewrites_fov_lines():
    out = adapt_commands(CFG, GridConfig(max_range=6, fov=90))
    assert out[0] == "sensorStop"
    assert "aoaFovCfg -1 -45 45 -150 150" in out
    assert "cfarFovCfg -1 0 0 6" in out
    assert "cfarFovCfg -1 1 -4.68 4.68" in out  # Doppler gate untouched
    assert out[-1] == "sensorStart"


def test_adapt_commands_requires_fov_lines():
    with pytest.raises(ValueError):
        adapt_commands(["flushCfg", "sensorStart"], GridConfig())


# ---------------------------------------------------------------- replay / files
def _write_recording(path, frames):
    with open(path, "w", encoding="utf-8") as fp:
        for pts, ts in frames:
            fp.write(" ".join(f"{v:.3f}" for p in pts for v in p) + f" {ts:.3f}\n")


def test_read_pointcloud_file(tmp_path):
    rec = tmp_path / "PointCloud.txt"
    _write_recording(rec, [(_pts((0.1, 1.5), (0.2, 2.5)), 100.0)])
    ((points, ts),) = list(read_pointcloud_file(rec))
    assert points.shape == (2, 6)
    assert ts == pytest.approx(100.0)


def test_replay_refills_gaps_with_empty_frames(tmp_path):
    rec = tmp_path / "PointCloud.txt"
    period = 1 / 30
    _write_recording(rec, [(_pts((0, 1.5)), 10.0), (_pts((0, 1.5)), 10.0 + 4 * period)])
    frames = list(replay_frames(rec, speed=0))
    assert [len(f.points) for f in frames] == [1, 0, 0, 0, 1]
    assert [f.header.frame_number for f in frames] == [1, 2, 3, 4, 5]


def test_zones_writer_format(tmp_path):
    out = tmp_path / "DetectedZones.txt"
    detected = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=bool)
    detected[1, 2] = True
    with DetectedZonesWriter(out) as w:
        w.write(detected, 12.5)
    values = out.read_text(encoding="utf-8").split()
    assert len(values) == 51
    assert values[-1] == "12.500"
    assert flatten_zones(detected) == [int(v) for v in values[:-1]]


def test_cli_replay_end_to_end(tmp_path, capsys):
    rec = tmp_path / "PointCloud.txt"
    period = 1 / 30
    _write_recording(rec, [(_pts(_polar(2.5, 30)), 5.0 + k * period) for k in range(6)])
    out_dir = tmp_path / "out"
    rc = main(
        ["--replay", str(rec), "--replay-speed", "0", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    lines = (out_dir / "DetectedZones.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    flags = [list(map(int, line.split()[:-1])) for line in lines]
    # ring 2 (2-3 m), sector 3 (+15..+45): row 7 from the top, column 3
    idx = (NUM_RINGS - 1 - 2) * NUM_SECTORS + 3
    assert [f[idx] for f in flags] == [0, 0, 0, 1, 1, 1]
    assert not (out_dir / "PointCloud.txt").exists()  # the input is not re-recorded
    out = capsys.readouterr().out
    assert "nearest 2-3 m, +15..+45 deg" in out
    assert "zone ON  2-3 m, +15..+45 deg" in out

    distances = (out_dir / "SectorDistances.txt").read_text(encoding="utf-8")
    rows = [line.split() for line in distances.splitlines()]
    assert len(rows) == 6
    assert rows[0][:5] == ["-1"] * 5  # nothing confirmed yet
    assert rows[3][3] == "2.500" and rows[3][2] == "-1"
    events = (out_dir / "ZoneEvents.txt").read_text(encoding="utf-8").split()
    assert events[1:] == ["ON", "2", "3"]


# ---------------------------------------------------------------- distances / events
def test_sector_distances_use_only_detected_zones():
    grid = GridConfig(max_range=6)
    detected = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=bool)
    detected[1, 2] = True  # 1-2 m, central sector
    pts = _pts(_polar(1.8, 0), _polar(1.3, 5), _polar(0.6, 0), _polar(3.0, 30))
    d = sector_distances(pts, detected, grid)
    # 0.6 m is in ring 0 (not detected) and 3.0 m in sector 3 (not detected)
    assert d[2] == pytest.approx(1.3)
    assert np.isnan(d[[0, 1, 3, 4]]).all()


def test_sector_distances_empty_frame():
    d = sector_distances(np.empty((0, 6)), np.ones((10, 5), dtype=bool), GridConfig())
    assert np.isnan(d).all()


def test_zone_events_on_and_off():
    prev = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=bool)
    cur = prev.copy()
    prev[4, 0] = True
    cur[1, 2] = True
    events = {(e.ring, e.sector, e.active) for e in zone_events(prev, cur)}
    assert events == {(4, 0, False), (1, 2, True)}


def test_zone_monitor_reports_events_once():
    monitor = ZoneMonitor(GridConfig(min_hits=2))
    target = _pts(_polar(1.5, 0))
    assert monitor.update(target).events == []
    first = monitor.update(target)
    assert [(e.ring, e.sector, e.active) for e in first.events] == [(1, 2, True)]
    assert first.distances[2] == pytest.approx(1.5)
    assert monitor.update(target).events == []  # still on: no new event
    off = monitor.update(np.empty((0, 6)))
    assert [(e.ring, e.sector, e.active) for e in off.events] == [(1, 2, False)]


def test_writers_and_console_format(tmp_path):
    d = np.array([np.nan, 2.3456, np.nan, 1.0, np.nan])
    with SectorDistancesWriter(tmp_path / "d.txt") as w:
        w.write(d, 7.0)
    assert (tmp_path / "d.txt").read_text().split() == [
        "-1",
        "2.346",
        "-1",
        "1.000",
        "-1",
        "7.000",
    ]
    assert format_distances(d) == "-  2.35  -  1.00  -"
    monitor = ZoneMonitor(GridConfig(min_hits=1))
    events = monitor.update(_pts(_polar(1.5, 0))).events
    with ZoneEventsWriter(tmp_path / "e.txt") as w:
        w.write(events, 9.5)
    assert (tmp_path / "e.txt").read_text() == "9.500 ON 1 2\n"


# ---------------------------------------------------------------- TX modes
def test_tx_mode_chirp_path(tmp_path):
    for n in (1, 2, 3):
        (tmp_path / f"short_range_radar_{n}tx.cfg").write_text("x")
    default = tmp_path / "short_range_radar_2tx.cfg"
    assert tx_mode_chirp_path(default, 3) == tmp_path / "short_range_radar_3tx.cfg"
    (tmp_path / "short_range_radar_3tx.cfg").unlink()
    with pytest.raises(FileNotFoundError):
        tx_mode_chirp_path(default, 3)


def test_tx_and_chirp_are_exclusive(capsys):
    with pytest.raises(SystemExit):
        main(["--config", "c.json", "--tx", "2", "--chirp", "x.cfg"])


# ---------------------------------------------------------------- viewer helpers
def test_proximity_levels_by_thirds():
    assert [proximity_level(r, 6) for r in range(6)] == [0, 0, 1, 1, 2, 2]


def test_zone_outline_clipped_to_max_range():
    xs, ys = zone_outline(3, 2, max_range=3.5)
    radii = np.hypot(xs, ys)
    assert radii.max() == pytest.approx(3.5)
    assert radii.min() == pytest.approx(3.0)

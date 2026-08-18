"""Unit tests for the point cloud and temperature file writers."""

from __future__ import annotations

import numpy as np

from urad_mmwave.parser import TemperatureReport
from urad_mmwave.writer import PointCloudWriter, TemperatureWriter


def test_pointcloud_format_matches_legacy(tmp_path):
    path = tmp_path / "out" / "PointCloud.txt"
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.5, 120, 30],
            [-1.5, 4.0, 0.25, -0.75, 95, 42],
        ]
    )
    with PointCloudWriter(path) as writer:
        writer.write(points, 1755500000.123)

    line = path.read_text(encoding="utf-8").strip()
    assert line == (
        "1.000 2.000 3.000 0.500 120 30 "
        "-1.500 4.000 0.250 -0.750 95 42 1755500000.123"
    )


def test_empty_frames_are_skipped(tmp_path):
    path = tmp_path / "PointCloud.txt"
    with PointCloudWriter(path) as writer:
        writer.write(np.zeros((0, 6)), 1755500000.0)
    assert path.read_text(encoding="utf-8") == ""


def test_temperature_format(tmp_path):
    path = tmp_path / "Temperature.txt"
    report = TemperatureReport(1, 5000, 40, 41, 42, 43, 50, 51, 52, 45, 60, 61)
    with TemperatureWriter(path) as writer:
        writer.write(report, 1755500000.5)
        writer.write(None, 1755500001.0)  # frames without TLV 9 are skipped

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["1 5000 40 41 42 43 50 51 52 45 60 61 1755500000.500"]

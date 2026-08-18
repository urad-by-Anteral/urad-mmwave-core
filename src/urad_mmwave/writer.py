"""File writers for radar output, compatible with the legacy uRAD format.

Point cloud lines: ``x y z v snr noise`` repeated per detected object,
followed by the host epoch timestamp. Temperature lines: the twelve TLV-9
fields followed by the timestamp. Files are kept open for the whole session
instead of being reopened on every frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from urad_mmwave.parser import TemperatureReport

log = logging.getLogger(__name__)


class _LineWriter:
    """Base append-only text writer with lazy directory creation."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # The file stays open for the whole session (closed via close/__exit__)
        # instead of being reopened on every frame.
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        log.info("Writing to %s", self._path)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


class PointCloudWriter(_LineWriter):
    """Appends one line per frame with all detected points and a timestamp."""

    def write(self, points: np.ndarray, timestamp: float) -> None:
        """Write one frame's point cloud; frames with no points are skipped."""
        if len(points) == 0:
            return
        parts = [
            f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {int(p[4])} {int(p[5])}"
            for p in points
        ]
        self._file.write(" ".join(parts) + f" {timestamp:.3f}\n")


class TemperatureWriter(_LineWriter):
    """Appends one line per temperature report (TLV type 9)."""

    def write(self, report: TemperatureReport | None, timestamp: float) -> None:
        if report is None:
            return
        self._file.write(
            f"{report.valid} {report.time_ms} "
            f"{report.rx0} {report.rx1} {report.rx2} {report.rx3} "
            f"{report.tx0} {report.tx1} {report.tx2} "
            f"{report.pm} {report.dig0} {report.dig1} {timestamp:.3f}\n"
        )

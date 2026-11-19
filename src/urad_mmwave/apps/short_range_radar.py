"""Short Range Radar application (uRAD Automotive and uRAD Automotive HPA).

Zone-based object detection on top of the standard **out-of-box** firmware
of each product (``out_of_box_1843_aop.bin`` for the uRAD Automotive,
``out_of_box_1843_isk.bin`` for the HPA): the radar streams its CFAR point
cloud and the host accumulates the points on a polar grid of 1 m range
rings (up to 10) by five 30° azimuth sectors covering ±75° around
boresight. A zone is
declared **detected** after it has been hit in ``min_hits`` consecutive
frames; one frame without points in it resets the count. This filters out
isolated false detections while reacting in a few frames (4 frames ≈ 0.13 s
at the 30 fps of the supplied chirp configuration).

Azimuth is measured from boresight (+Y), positive towards +X (the right of
an observer standing behind the radar), so sector 0 spans −75…−45° and
sector 4 spans +45…+75°.

The chirp configuration is a regular out-of-box ``.cfg`` file, one per
product (in the ``applications/short_range_radar`` folder of each product
repository); the application rewrites its field-of-view lines
(``aoaFovCfg`` and the range ``cfarFovCfg``) from ``--max-range`` and
``--fov`` so the firmware only reports points inside the monitored area.

Recorded ``PointCloud.txt`` files can be replayed without hardware
(``--replay``), which is also how the zone logic is tested.

Typical usage:

    urad-short-range-radar --config config_radar.json --gui
    urad-short-range-radar --config config_radar.json --max-range 6 --fov 90
    urad-short-range-radar --replay output/PointCloud.txt --gui
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep, time

import numpy as np

from urad_mmwave import __version__
from urad_mmwave.config import load_config
from urad_mmwave.parser import Frame, FrameHeader
from urad_mmwave.radar import RadarSession, read_chirp_config
from urad_mmwave.writer import PointCloudWriter, _LineWriter

log = logging.getLogger(__name__)

NUM_RINGS = 10
"""Number of 1 m range rings of the detection grid (0-1 m … 9-10 m)."""
NUM_SECTORS = 5
"""Number of azimuth sectors of the detection grid."""
SECTOR_WIDTH_DEG = 30.0
GRID_HALF_SPAN_DEG = NUM_SECTORS * SECTOR_WIDTH_DEG / 2  # 75°

FOV_CHOICES = (30, 90, 150)
"""Supported fields of view (degrees): one, three or all five sectors."""
DEFAULT_MAX_RANGE = 6
DEFAULT_FOV = 150
DEFAULT_MIN_HITS = 4
NOMINAL_FRAME_PERIOD_S = 1 / 30


# --------------------------------------------------------------------------
# Detection grid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GridConfig:
    """Monitored area and alarm filter.

    Attributes:
        max_range: Maximum detection range in meters (1-10).
        fov: Field of view in degrees, centered on boresight (30, 90 or 150).
        min_hits: Consecutive frames a zone must be hit to be detected.
    """

    max_range: float = DEFAULT_MAX_RANGE
    fov: int = DEFAULT_FOV
    min_hits: int = DEFAULT_MIN_HITS

    def __post_init__(self) -> None:
        if not 0 < self.max_range <= NUM_RINGS:
            raise ValueError(
                f"max_range must be in (0, {NUM_RINGS}] m, got {self.max_range}"
            )
        if self.fov not in FOV_CHOICES:
            raise ValueError(f"fov must be one of {FOV_CHOICES}, got {self.fov}")
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")

    @property
    def num_rings(self) -> int:
        """Rings covered by ``max_range`` (a partial last ring counts)."""
        return math.ceil(self.max_range)

    @property
    def sectors(self) -> range:
        """Indices of the sectors inside the field of view."""
        n = self.fov // int(SECTOR_WIDTH_DEG)
        first = (NUM_SECTORS - n) // 2
        return range(first, first + n)


def to_polar(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Range (m) and azimuth (degrees from boresight, + towards +X) of the
    points' X/Y components. ``points`` has x, y in its first two columns."""
    if len(points) == 0:
        return np.empty(0), np.empty(0)
    x, y = points[:, 0], points[:, 1]
    return np.hypot(x, y), np.degrees(np.arctan2(x, y))


def zone_indices(
    points: np.ndarray, grid: GridConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points to grid cells.

    Returns ``(rings, sectors, inside)``: the ring and sector index of every
    point and a boolean mask of the points inside the monitored area (in
    front of the radar, closer than ``max_range`` and within the field of
    view). Indices of points outside the area are meaningless.
    """
    rng, az = to_polar(points)
    if len(rng) == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=bool)
    half_fov = grid.fov / 2
    inside = (points[:, 1] > 0) & (rng < grid.max_range) & (np.abs(az) <= half_fov)
    rings = np.clip(np.floor(rng).astype(int), 0, NUM_RINGS - 1)
    sectors = np.clip(
        np.floor((az + GRID_HALF_SPAN_DEG) / SECTOR_WIDTH_DEG).astype(int),
        0,
        NUM_SECTORS - 1,
    )
    return rings, sectors, inside


class ZoneDetector:
    """Consecutive-hit filter over the ring × sector grid.

    ``update()`` takes one frame's point cloud and returns the boolean
    ``(NUM_RINGS, NUM_SECTORS)`` map of detected zones (ring 0 = nearest).
    """

    def __init__(self, grid: GridConfig):
        self.grid = grid
        self.hits = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=int)

    def update(self, points: np.ndarray) -> np.ndarray:
        hit = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=bool)
        rings, sectors, inside = zone_indices(points, self.grid)
        hit[rings[inside], sectors[inside]] = True
        self.hits[hit] += 1
        self.hits[~hit] = 0
        return self.hits >= self.grid.min_hits

    def reset(self) -> None:
        self.hits[:] = 0


def zone_label(ring: int, sector: int) -> str:
    """Human readable zone name, e.g. ``'2-3 m, -15..+15 deg'``."""
    low = -GRID_HALF_SPAN_DEG + sector * SECTOR_WIDTH_DEG
    return f"{ring}-{ring + 1} m, {low:+.0f}..{low + SECTOR_WIDTH_DEG:+.0f} deg"


def flatten_zones(detected: np.ndarray) -> list[int]:
    """Flatten the zone map in the ``DetectedZones.txt`` order: rows from the
    farthest ring (9-10 m) to the nearest (0-1 m), sectors left to right."""
    return [int(v) for v in detected[::-1].ravel()]


# --------------------------------------------------------------------------
# Chirp configuration
# --------------------------------------------------------------------------
def adapt_commands(commands: Iterable[str], grid: GridConfig) -> list[str]:
    """Rewrite the field-of-view lines of an out-of-box configuration.

    ``aoaFovCfg`` limits the azimuth to ±fov/2 and the range ``cfarFovCfg``
    (second argument 0) to ``max_range``, so the firmware drops points
    outside the monitored area. A leading ``sensorStop`` is added if missing
    so a running radar can be reconfigured without a reset.
    """
    half = grid.fov / 2
    out = []
    seen_aoa = seen_range = False
    for line in commands:
        parts = line.split()
        if parts and parts[0] == "aoaFovCfg" and len(parts) >= 6:
            parts[2], parts[3] = f"{-half:g}", f"{half:g}"
            line, seen_aoa = " ".join(parts), True
        elif parts and parts[0] == "cfarFovCfg" and len(parts) >= 5 and parts[2] == "0":
            parts[3], parts[4] = "0", f"{grid.max_range:g}"
            line, seen_range = " ".join(parts), True
        out.append(line)
    if not (seen_aoa and seen_range):
        raise ValueError(
            "The chirp configuration must contain 'aoaFovCfg' and a range "
            "'cfarFovCfg <subframe> 0 ...' line"
        )
    if not out or out[0] != "sensorStop":
        out.insert(0, "sensorStop")
    return out


# --------------------------------------------------------------------------
# Replay of recorded point clouds
# --------------------------------------------------------------------------
def read_pointcloud_file(path: str | Path) -> Iterator[tuple[np.ndarray, float]]:
    """Yield ``(points, timestamp)`` from a ``PointCloud.txt`` file
    (``x y z v snr noise`` per point, then the timestamp, one frame per
    line). ``points`` has shape (n, 6)."""
    with open(path, encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, 1):
            values = line.split()
            if not values:
                continue
            if (len(values) - 1) % 6:
                log.warning("%s:%d: malformed line skipped", path, lineno)
                continue
            data = np.array(values, dtype=float)
            yield data[:-1].reshape(-1, 6), float(data[-1])


def replay_frames(
    path: str | Path, speed: float = 1.0, frame_period: float = NOMINAL_FRAME_PERIOD_S
) -> Iterator[Frame]:
    """Turn a recorded point cloud file into frames, paced in real time.

    The writers skip frames without points, so gaps longer than one frame
    period are refilled with empty frames — otherwise a zone would stay
    detected across the gap. ``speed`` scales the pace (0 = as fast as
    possible).
    """
    previous_ts = None
    start_wall = time()
    start_ts = None
    frame_number = 0

    def make(points: np.ndarray, ts: float) -> Frame:
        nonlocal frame_number
        frame_number += 1
        header = FrameHeader(0, 0, 0, frame_number, 0, len(points), 0, 0)
        return Frame(header=header, points=points, timestamp=ts)

    for points, ts in read_pointcloud_file(path):
        if start_ts is None:
            start_ts = ts
        if previous_ts is not None:
            missing = round((ts - previous_ts) / frame_period) - 1
            for k in range(1, min(missing, 300) + 1):
                yield make(np.empty((0, 6)), previous_ts + k * frame_period)
        if speed > 0:
            delay = (ts - start_ts) / speed - (time() - start_wall)
            if delay > 0:
                sleep(delay)
        yield make(points, ts)
        previous_ts = ts


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
class DetectedZonesWriter(_LineWriter):
    """Appends the 50 zone flags (see :func:`flatten_zones`) and the
    timestamp, one line per frame."""

    def write(self, detected: np.ndarray, timestamp: float) -> None:
        flags = " ".join(str(v) for v in flatten_zones(detected))
        self._file.write(f"{flags} {timestamp:.3f}\n")


@dataclass
class _Stats:
    frames: int = 0
    alarms: int = 0
    started: float = field(default_factory=time)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-short-range-radar",
        description="Zone-based short range object detection with a uRAD "
        "Automotive or uRAD Automotive HPA radar running the out-of-box "
        "firmware.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-c", "--config", help="Path to the JSON configuration file (radar mode)"
    )
    source.add_argument(
        "--replay",
        metavar="FILE",
        help="Replay a recorded PointCloud.txt instead of reading the radar",
    )
    parser.add_argument("--control-port", help="Override the control serial port")
    parser.add_argument("--data-port", help="Override the data serial port")
    parser.add_argument(
        "--single-port",
        metavar="PORT",
        help="Use one shared serial port for control and data "
        "(e.g. /dev/serial0 on Raspberry Pi)",
    )
    parser.add_argument("--chirp", help="Override the chirp configuration file path")
    parser.add_argument(
        "--max-range",
        type=float,
        default=DEFAULT_MAX_RANGE,
        metavar="METERS",
        help=f"Maximum detection range, up to {NUM_RINGS} m "
        f"(default: {DEFAULT_MAX_RANGE})",
    )
    parser.add_argument(
        "--fov",
        type=int,
        choices=FOV_CHOICES,
        default=DEFAULT_FOV,
        help=f"Field of view in degrees (default: {DEFAULT_FOV})",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=DEFAULT_MIN_HITS,
        metavar="N",
        help="Consecutive frames a zone must be hit to raise its alarm "
        f"(default: {DEFAULT_MIN_HITS})",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        metavar="X",
        help="Replay pace relative to real time; 0 = as fast as possible "
        "(default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for PointCloud.txt and DetectedZones.txt "
        "(default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--print-points", action="store_true", help="Print every detected point"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live top view with the detection grid and zone alarms "
        "(requires: pip install urad-mmwave[gui])",
    )
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        metavar="N",
        help="Stop after this many frames (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--gpio-reset-pin",
        type=int,
        metavar="PIN",
        help="Reset the radar through this GPIO pin before configuring "
        "(Raspberry Pi setups; requires: pip install urad-mmwave[rpi])",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        grid = GridConfig(
            max_range=args.max_range, fov=args.fov, min_hits=args.min_hits
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    detector = ZoneDetector(grid)
    stats = _Stats()
    output_dir = Path(args.output_dir)

    try:
        with ExitStack() as stack:
            if args.replay:
                frames: Iterator[Frame] = replay_frames(args.replay, args.replay_speed)
                log.info("Replaying %s", args.replay)
            else:
                try:
                    config = load_config(args.config)
                except (FileNotFoundError, ValueError) as exc:
                    log.error("%s", exc)
                    return 1
                if args.control_port:
                    config.control_serial.port = args.control_port
                if args.data_port:
                    config.data_serial.port = args.data_port
                if args.single_port:
                    config.control_serial.port = args.single_port
                    config.data_serial.port = args.single_port
                if args.chirp:
                    config.chirp_config_path = args.chirp
                if args.gpio_reset_pin is not None:
                    config.gpio_reset_pin = args.gpio_reset_pin
                commands = adapt_commands(
                    read_chirp_config(config.chirp_config_path), grid
                )
                session = stack.enter_context(RadarSession(config, commands=commands))
                frames = session.frames()

            pointcloud_writer = zones_writer = None
            if not args.no_save:
                # in replay mode the input file is not re-recorded
                if not args.replay:
                    pointcloud_writer = stack.enter_context(
                        PointCloudWriter(output_dir / "PointCloud.txt")
                    )
                zones_writer = stack.enter_context(
                    DetectedZonesWriter(output_dir / "DetectedZones.txt")
                )

            def process(frame: Frame) -> np.ndarray:
                detected = detector.update(frame.points)
                stats.frames += 1
                active = np.argwhere(detected)
                line = f"frame {frame.header.frame_number}: {len(frame.points)} points"
                if len(active):
                    stats.alarms += 1
                    nearest = zone_label(*active[np.argmin(active[:, 0])])
                    line += f", {len(active)} zone(s) detected, nearest {nearest}"
                print(line)
                if args.print_points:
                    for p in frame.points:
                        print(
                            f"  x: {p[0]:.3f} m, y: {p[1]:.3f} m, z: {p[2]:.3f} m, "
                            f"v: {p[3]:.3f} m/s, snr: {int(p[4])}, noise: {int(p[5])}"
                        )
                if pointcloud_writer is not None:
                    pointcloud_writer.write(frame.points, frame.timestamp)
                if zones_writer is not None:
                    zones_writer.write(detected, frame.timestamp)
                return detected

            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration and --max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.short_range_radar_viewer import run_viewer

                run_viewer(frames, grid, process)
            else:
                for frame in frames:
                    process(frame)
                    if args.max_frames is not None and stats.frames >= args.max_frames:
                        log.info("Reached %d frames; stopping", args.max_frames)
                        break
                    if (
                        args.duration is not None
                        and time() - stats.started >= args.duration
                    ):
                        log.info("Reached %.1f s; stopping", args.duration)
                        break
    except KeyboardInterrupt:
        log.info("Interrupted by user; stopping")
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of a traceback
        log.error("%s", exc)
        return 1

    log.info(
        "Processed %d frames (%d with zone alarms) in %.1f s",
        stats.frames,
        stats.alarms,
        time() - stats.started,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

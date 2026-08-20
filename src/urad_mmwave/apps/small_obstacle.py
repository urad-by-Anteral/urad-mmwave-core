"""Small Obstacle Detection application (uRAD Industrial).

Requires the TI Small Obstacle Detection firmware (Radar Toolbox, Robotics
family), built for mobile robots — e.g. robotic lawn mowers — that must
detect small, low obstacles near the ground (rocks, apples, hedgehogs)
before driving over them. The firmware combines the 3D People Tracking
processing chain (with enhanced static detection and a persistent point
cloud fed to the tracker) with an occupancy state machine that raises a
per-zone alarm based on the number of points and their average SNR inside
each configured zone.

The firmware shares the out-of-box packet framing (same sync word and
header) and the people tracking TLV set, plus one additional TLV:

    1010 — target list (tracked obstacles with position/velocity/acceleration)
    1011 — target index (track id per point of the previous frame)
    1020 — compressed spherical point cloud
    1021 — presence indication
    1030 — zone occupancy bit mask (bit N set = zone N occupied)

Zones are configured in the chirp configuration file with the ``zoneDef``
command (up to 2 zones on the IWR6843) and the state machine thresholds
with ``occStateMach``.

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.small_obstacle import parse_frame

    with RadarSession(load_config("config_radar.json")) as session:
        for fields, payload, timestamp in session.packets():
            frame = parse_frame(payload, timestamp)
            if frame.occupied_zones():
                print("obstacle in zones", frame.occupied_zones())
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import numpy as np

from urad_mmwave import __version__
from urad_mmwave.apps.people_tracking import (
    _PRESENCE_STRUCT,
    _TLV_HEADER,
    MAX_TLV_LENGTH,
    MAX_TLV_TYPE,
    PADDING_WORD,
    TLV_POINT_CLOUD,
    TLV_PRESENCE,
    TLV_TARGET_HEIGHT,
    TLV_TARGET_INDEX,
    TLV_TARGET_LIST,
    PeopleTrackingFrame,
    Target,
    _AppendWriter,
    _parse_heights,
    _parse_point_cloud,
    _parse_targets,
)
from urad_mmwave.config import load_config
from urad_mmwave.radar import RadarSession

log = logging.getLogger(__name__)

TLV_OCCUPANCY = 1030

# The 6843 firmware supports at most 2 zones (MMWDEMO_MAX_ZONES); the TLV
# itself is a 32-bit mask, so decode all 32 bits and let the configuration
# decide how many are meaningful.
MAX_ZONES = 32

ZONE_COMMAND = "zoneDef"

_OCCUPANCY_STRUCT = struct.Struct("<I")


@dataclass
class SmallObstacleFrame:
    """One decoded small obstacle detection frame.

    Attributes:
        tracking: The people tracking part of the frame (point cloud,
            tracked obstacles, target index, heights, presence).
        occupancy: Zone occupancy bit mask from TLV 1030 (bit N set means
            zone N is occupied), or ``None`` if the frame carried no
            occupancy TLV.
        timestamp: Host epoch time when the frame was received.
    """

    tracking: PeopleTrackingFrame = field(default_factory=PeopleTrackingFrame)
    occupancy: int | None = None
    timestamp: float = 0.0

    @property
    def points(self) -> np.ndarray:
        """Point cloud of shape (N, 5): range, azimuth°, elevation°, doppler, snr."""
        return self.tracking.points

    @property
    def obstacles(self) -> list[Target]:
        """Tracked obstacles (position/velocity/acceleration per track)."""
        return self.tracking.targets

    def zone_occupied(self, zone: int) -> bool:
        """True if the given zone index is flagged occupied in this frame."""
        if self.occupancy is None or not 0 <= zone < MAX_ZONES:
            return False
        return bool(self.occupancy >> zone & 1)

    def occupied_zones(self) -> list[int]:
        """Indices of all zones flagged occupied in this frame."""
        if not self.occupancy:
            return []
        return [zone for zone in range(MAX_ZONES) if self.occupancy >> zone & 1]


def points_to_xyz(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project spherical points (range, azimuth°, elevation°, ...) to x/y/z meters.

    Coordinates are radar-relative: x right, y forward, z up (a point below
    the radar — e.g. near the ground for an elevated sensor — has z < 0).
    """
    if not len(points):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    range_m = points[:, 0]
    azimuth = np.radians(points[:, 1])
    elevation = np.radians(points[:, 2])
    return (
        range_m * np.cos(elevation) * np.sin(azimuth),
        range_m * np.cos(elevation) * np.cos(azimuth),
        range_m * np.sin(elevation),
    )


def read_zones(chirp_config_path: str | Path) -> dict[int, tuple[float, ...]]:
    """Extract the occupancy zones from a chirp configuration file.

    Returns a mapping of zone index to ``(xmin, xmax, ymin, ymax, zmin,
    zmax)`` for each ``zoneDef`` command present in the file.
    """
    from urad_mmwave.radar import read_chirp_config

    zones: dict[int, tuple[float, ...]] = {}
    for command in read_chirp_config(chirp_config_path):
        parts = command.split()
        if parts[0] == ZONE_COMMAND and len(parts) == 8:
            zones[int(parts[1])] = tuple(float(value) for value in parts[2:])
    return zones


def parse_frame(payload: bytes, timestamp: float = 0.0) -> SmallObstacleFrame:
    """Decode the TLV payload of one small obstacle detection packet."""
    tracking = PeopleTrackingFrame(timestamp=timestamp)
    frame = SmallObstacleFrame(tracking=tracking, timestamp=timestamp)
    cursor = 0

    while cursor + _TLV_HEADER.size <= len(payload):
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

        if tlv_type == PADDING_WORD:
            break  # end-of-frame alignment padding
        if tlv_type > MAX_TLV_TYPE or tlv_length > MAX_TLV_LENGTH:
            log.warning(
                "Implausible TLV (type=%d, length=%d); discarding rest of frame",
                tlv_type,
                tlv_length,
            )
            break
        if cursor + tlv_length > len(payload):
            log.warning(
                "TLV type %d declares %d bytes but only %d remain; discarding",
                tlv_type,
                tlv_length,
                len(payload) - cursor,
            )
            break

        body = payload[cursor : cursor + tlv_length]

        if tlv_type == TLV_POINT_CLOUD:
            tracking.points = _parse_point_cloud(body)
        elif tlv_type == TLV_TARGET_LIST:
            tracking.targets = _parse_targets(body)
        elif tlv_type == TLV_TARGET_INDEX:
            tracking.target_index = np.frombuffer(body, dtype=np.uint8).astype(int)
        elif tlv_type == TLV_TARGET_HEIGHT:
            tracking.heights = _parse_heights(body)
        elif tlv_type == TLV_PRESENCE:
            if len(body) >= _PRESENCE_STRUCT.size:
                tracking.presence = _PRESENCE_STRUCT.unpack_from(body, 0)[0]
        elif tlv_type == TLV_OCCUPANCY:
            if len(body) >= _OCCUPANCY_STRUCT.size:
                frame.occupancy = _OCCUPANCY_STRUCT.unpack_from(body, 0)[0]
            else:
                log.warning("Occupancy TLV too short (%d bytes)", len(body))
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length

    return frame


def format_zone_status(frame: SmallObstacleFrame, num_zones: int) -> str:
    """One-line zone status for console output, e.g. ``0:OCCUPIED 1:clear``."""
    if frame.occupancy is None:
        return "n/a"
    # Show at least the configured zones, plus any unexpected extra bits.
    shown = max(num_zones, max(frame.occupied_zones(), default=-1) + 1, 1)
    return " ".join(
        f"{zone}:{'OCCUPIED' if frame.zone_occupied(zone) else 'clear'}"
        for zone in range(shown)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-small-obstacle",
        description="Small obstacle detection with a uRAD Industrial radar "
        "running the Small Obstacle Detection firmware (zone occupancy "
        "alarms for mobile robots).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the JSON configuration file (serial ports and chirp path)",
    )
    parser.add_argument("--control-port", help="Override the control serial port")
    parser.add_argument("--data-port", help="Override the data serial port")
    parser.add_argument(
        "--single-port",
        metavar="PORT",
        help="Use one physical UART for control and data "
        "(e.g. /dev/serial0 on Raspberry Pi)",
    )
    parser.add_argument("--chirp", help="Override the chirp configuration file path")
    parser.add_argument(
        "--gpio-reset-pin",
        type=int,
        metavar="PIN",
        help="BCM pin to reset the chip before configuring (Raspberry Pi)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for PointCloud/Obstacles/ZoneOccupancy files "
        "(default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live top view with the height-colored point cloud, "
        "tracked obstacles and occupancy zones "
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
        help="Stop after N frames (default: unlimited)",
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
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    if args.single_port:
        config.control_serial.port = args.single_port
        config.data_serial.port = args.single_port
    if args.control_port:
        config.control_serial.port = args.control_port
    if args.data_port:
        config.data_serial.port = args.data_port
    if args.chirp:
        config.chirp_config_path = args.chirp
    if args.gpio_reset_pin is not None:
        config.gpio_reset_pin = args.gpio_reset_pin

    output_dir = Path(args.output_dir)
    start_time = time()
    frame_count = 0

    try:
        num_zones = len(read_zones(config.chirp_config_path))
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    if num_zones == 0:
        log.warning(
            "No zoneDef commands in %s; the firmware will report no "
            "occupancy alarms",
            config.chirp_config_path,
        )

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            writers = {}
            if not args.no_save:
                for name in ("PointCloud", "Obstacles", "ZoneOccupancy"):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def handle_frame(frame: SmallObstacleFrame) -> None:
                nonlocal frame_count
                frame_count += 1
                print(
                    f"zones: {format_zone_status(frame, num_zones)}  "
                    f"obstacles: {len(frame.obstacles)}  "
                    f"points: {len(frame.points)}"
                )

                if writers:
                    writers["PointCloud"].write_row(
                        [v for p in frame.points for v in p], frame.timestamp
                    )
                    writers["Obstacles"].write_row(
                        [
                            v
                            for t in frame.obstacles
                            for v in (t.tid, *t.position, *t.velocity, *t.acceleration)
                        ],
                        frame.timestamp,
                    )
                    if frame.occupancy is not None:
                        writers["ZoneOccupancy"].write_row(
                            [frame.occupancy], frame.timestamp
                        )

            if args.gui:
                if args.max_frames is not None or args.duration is not None:
                    log.warning(
                        "--max-frames/--duration are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.small_obstacle_viewer import run_viewer

                run_viewer(
                    config,
                    (
                        parse_frame(payload, timestamp)
                        for _fields, payload, timestamp in session.packets()
                    ),
                    on_frame=handle_frame,
                )
            else:
                for _fields, payload, timestamp in session.packets():
                    frame = parse_frame(payload, timestamp)
                    handle_frame(frame)
                    if args.max_frames is not None and frame_count >= args.max_frames:
                        log.info("Reached %d frames; stopping", args.max_frames)
                        break
                    if (
                        args.duration is not None
                        and time() - start_time >= args.duration
                    ):
                        log.info("Reached %.1f s; stopping", args.duration)
                        break
    except KeyboardInterrupt:
        log.info("Interrupted by user; stopping sensor")
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of a traceback
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

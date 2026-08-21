"""Area Scanner application (uRAD Industrial).

Requires the TI Area Scanner firmware (``area_scanner_68xx_demo_aop.bin``,
distributed as a release asset in the uRAD Industrial repository). The
firmware combines the out-of-box moving-object detection chain with a
static object detection algorithm and the group tracker (GTRACK), so each
frame reports moving objects, *newly added* static objects (boxes, carts,
pallets left in the scene after the 15-frame calibration) and tracked
objects with position, velocity and acceleration.

The packet framing matches the out-of-box demo except for the frame
header, which appends a tenth word with the number of static detected
objects (44 bytes total, :data:`HEADER_FORMAT`). The TLV set is:

    1  — dynamic point cloud (range, azimuth, elevation in spherical
         coordinates, doppler)
    7  — dynamic point cloud side info (snr, noise)
    8  — static point cloud (x, y, z cartesian, doppler)
    9  — static point cloud side info (snr, noise)
    10 — tracked object list (tid, position/velocity/acceleration)
    11 — point-to-track association (track id per dynamic point)

Zone occupancy (critical/warning zones by radial distance, with velocity
projection) is not computed on the device; this module reimplements the
logic of the TI Area Scanner visualizer in :func:`classify_zones`.

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.area_scanner import HEADER_FORMAT, parse_frame

    config = load_config("config_radar.json")
    config.packet.header_format = HEADER_FORMAT
    with RadarSession(config) as session:
        for fields, payload, timestamp in session.packets():
            frame = parse_frame(payload, timestamp, num_tlvs=fields[6])
            print(len(frame.targets), "objects tracked")
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
    _TLV_HEADER,
    MAX_TLV_LENGTH,
    MAX_TLV_TYPE,
    PADDING_WORD,
    _AppendWriter,
    points_to_xy,  # noqa: F401 - re-exported for the viewer and library users
    read_boundary_boxes,  # noqa: F401 - same zone commands as people tracking
)
from urad_mmwave.config import load_config
from urad_mmwave.radar import RadarSession

log = logging.getLogger(__name__)

TLV_DYNAMIC_POINTS = 1
TLV_DYNAMIC_SIDE_INFO = 7
TLV_STATIC_POINTS = 8
TLV_STATIC_SIDE_INFO = 9
TLV_TRACK_LIST = 10
TLV_TARGET_INDEX = 11

# Area Scanner frame header: the out-of-box header plus a trailing
# numStaticDetectedObj word (44 bytes). Field order after the sync word:
# version, totalPacketLen, platform, frameNumber, timeCpuCycles,
# numDetectedObj, numTLVs, subFrameNumber, numStaticDetectedObj.
HEADER_FORMAT = "<Q9I"
_HEADER_NUM_TLVS_INDEX = 6  # position of numTLVs in the fields after the sync

# Point-to-track association codes (TLV 11) for unassociated points.
TRACK_INDEX_WEAK_SNR = 253  # does not meet the SNR requirements
TRACK_INDEX_OUT_OF_BOUNDS = 254  # outside the boundaryBox area
TRACK_INDEX_NOISE = 255  # considered noise

_DYNAMIC_POINT_STRUCT = struct.Struct("<4f")  # range, azimuth, elevation, doppler
_STATIC_POINT_STRUCT = struct.Struct("<4f")  # x, y, z, doppler
_SIDE_INFO_STRUCT = struct.Struct("<2H")  # snr, noise
# tid, posX, posY, velX, velY, accX, accY, posZ, velZ, accZ
_TARGET_STRUCT = struct.Struct("<I9f")


@dataclass(frozen=True)
class AreaTarget:
    """One tracked object from the track list TLV (type 10)."""

    tid: int
    position: tuple[float, float, float]  # x, y, z in meters
    velocity: tuple[float, float, float]  # m/s
    acceleration: tuple[float, float, float]  # m/s^2


@dataclass(frozen=True)
class ZoneConfig:
    """Radial occupancy zones around the sensor (TI visualizer defaults)."""

    critical: tuple[float, float] = (0.0, 2.0)  # start, end in meters
    warning: tuple[float, float] = (2.0, 4.0)  # start, end in meters
    projection_time: float = 2.0  # seconds ahead for the warning projection


@dataclass(frozen=True)
class ZoneStatus:
    """Occupancy result of one frame against a :class:`ZoneConfig`."""

    critical: bool
    warning: bool

    @property
    def label(self) -> str:
        if self.critical:
            return "CRITICAL"
        if self.warning:
            return "warning"
        return "clear"


@dataclass
class AreaScannerFrame:
    """One decoded area scanner frame.

    Attributes:
        dynamic_points: Array of shape (N, 6) with columns range (m),
            azimuth (deg), elevation (deg), doppler (m/s), snr, noise.
        static_points: Array of shape (M, 6) with columns x, y, z (m),
            doppler (m/s), snr, noise.
        targets: Tracked objects.
        target_index: Track id per dynamic point (codes >= 250 mean the
            point was not associated, see ``TRACK_INDEX_*``).
        timestamp: Host epoch time when the frame was received.
    """

    dynamic_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    static_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    targets: list[AreaTarget] = field(default_factory=list)
    target_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    timestamp: float = 0.0


def _parse_point_block(body: bytes, point_struct: struct.Struct) -> np.ndarray:
    """Unpack an array of 4-float points (dynamic or static cloud)."""
    count = len(body) // point_struct.size
    points = np.zeros((count, 4))
    for j in range(count):
        points[j] = point_struct.unpack_from(body, j * point_struct.size)
    return points


def _parse_side_info(body: bytes) -> np.ndarray:
    count = len(body) // _SIDE_INFO_STRUCT.size
    info = np.zeros((count, 2))
    for j in range(count):
        info[j] = _SIDE_INFO_STRUCT.unpack_from(body, j * _SIDE_INFO_STRUCT.size)
    return info


def _parse_targets(body: bytes) -> list[AreaTarget]:
    targets = []
    for j in range(len(body) // _TARGET_STRUCT.size):
        tid, px, py, vx, vy, ax, ay, pz, vz, az = _TARGET_STRUCT.unpack_from(
            body, j * _TARGET_STRUCT.size
        )
        targets.append(
            AreaTarget(
                tid=tid,
                position=(px, py, pz),
                velocity=(vx, vy, vz),
                acceleration=(ax, ay, az),
            )
        )
    return targets


def _with_side_info(points: np.ndarray, side_info: np.ndarray) -> np.ndarray:
    """Append snr/noise columns to a (N, 4) point array (zeros if absent)."""
    combined = np.zeros((len(points), 6))
    combined[:, :4] = points
    count = min(len(points), len(side_info))
    combined[:count, 4:6] = side_info[:count]
    return combined


def parse_frame(
    payload: bytes, timestamp: float = 0.0, num_tlvs: int | None = None
) -> AreaScannerFrame:
    """Decode the TLV payload of one area scanner packet.

    Args:
        payload: Packet bytes after the 44-byte frame header.
        timestamp: Host epoch time to attach to the frame.
        num_tlvs: TLV count from the frame header. Without it the payload
            is scanned until the zero-byte alignment padding is reached.
    """
    frame = AreaScannerFrame(timestamp=timestamp)
    dynamic = np.zeros((0, 4))
    dynamic_side = np.zeros((0, 2))
    static = np.zeros((0, 4))
    static_side = np.zeros((0, 2))
    cursor = 0
    remaining = num_tlvs

    while cursor + _TLV_HEADER.size <= len(payload):
        if remaining is not None and remaining <= 0:
            break
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

        if tlv_type in (0, PADDING_WORD):
            break  # end-of-frame alignment padding (zero bytes)
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

        if tlv_type == TLV_DYNAMIC_POINTS:
            dynamic = _parse_point_block(body, _DYNAMIC_POINT_STRUCT)
            # Spherical angles arrive in radians; store degrees like the
            # other uRAD applications.
            dynamic[:, 1:3] = np.degrees(dynamic[:, 1:3])
        elif tlv_type == TLV_DYNAMIC_SIDE_INFO:
            dynamic_side = _parse_side_info(body)
        elif tlv_type == TLV_STATIC_POINTS:
            static = _parse_point_block(body, _STATIC_POINT_STRUCT)
        elif tlv_type == TLV_STATIC_SIDE_INFO:
            static_side = _parse_side_info(body)
        elif tlv_type == TLV_TRACK_LIST:
            frame.targets = _parse_targets(body)
        elif tlv_type == TLV_TARGET_INDEX:
            frame.target_index = np.frombuffer(body, dtype=np.uint8).astype(int)
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length
        if remaining is not None:
            remaining -= 1

    frame.dynamic_points = _with_side_info(dynamic, dynamic_side)
    frame.static_points = _with_side_info(static, static_side)
    return frame


def projected_position(
    target: AreaTarget, projection_time: float
) -> tuple[float, float, float]:
    """Position of a track ``projection_time`` seconds ahead (constant velocity)."""
    return (
        target.position[0] + target.velocity[0] * projection_time,
        target.position[1] + target.velocity[1] * projection_time,
        target.position[2] + target.velocity[2] * projection_time,
    )


def zone_of(radial: float, zones: ZoneConfig) -> str:
    """Name the zone (``critical``/``warning``/``clear``) of a radial distance."""
    if zones.critical[0] <= radial <= zones.critical[1]:
        return "critical"
    if zones.warning[0] <= radial <= zones.warning[1]:
        return "warning"
    return "clear"


def classify_zones(frame: AreaScannerFrame, zones: ZoneConfig) -> ZoneStatus:
    """Evaluate the occupancy zones the way the TI Area Scanner visualizer does.

    The critical zone triggers on the presence of *any* detection — a
    dynamic point, a newly added static point or a tracked object — within
    its radial range. The warning zone triggers only on tracked objects
    whose current or projected position (``projection_time`` seconds ahead)
    falls within its radial range; static points outside the critical zone
    never trigger it.
    """
    critical = False
    warning = False

    if len(frame.dynamic_points):
        radial = frame.dynamic_points[:, 0]  # range is the radial distance
        critical |= bool(
            np.any((radial >= zones.critical[0]) & (radial <= zones.critical[1]))
        )
    if len(frame.static_points):
        radial = np.linalg.norm(frame.static_points[:, :3], axis=1)
        critical |= bool(
            np.any((radial >= zones.critical[0]) & (radial <= zones.critical[1]))
        )

    for target in frame.targets:
        current = float(np.linalg.norm(target.position))
        projected = float(
            np.linalg.norm(projected_position(target, zones.projection_time))
        )
        if zones.critical[0] <= current <= zones.critical[1]:
            critical = True
        if any(
            zones.warning[0] <= radial <= zones.warning[1]
            for radial in (current, projected)
        ):
            warning = True

    return ZoneStatus(critical=critical, warning=warning)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-area-scanner",
        description="Area scanning with a uRAD Industrial radar running the "
        "Area Scanner firmware: moving object tracking, newly added static "
        "object detection and occupancy zone monitoring.",
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
        help="Use one shared serial port for control and data "
        "(e.g. Raspberry Pi single-UART adapter)",
    )
    parser.add_argument("--chirp", help="Override the chirp configuration file path")
    parser.add_argument(
        "--critical-zone",
        type=float,
        nargs=2,
        default=(0.0, 2.0),
        metavar=("START", "END"),
        help="Critical zone radial range in meters; any detection inside "
        "triggers it (default: 0 2)",
    )
    parser.add_argument(
        "--warning-zone",
        type=float,
        nargs=2,
        default=(2.0, 4.0),
        metavar=("START", "END"),
        help="Warning zone radial range in meters; a track's current or "
        "projected position inside triggers it (default: 2 4)",
    )
    parser.add_argument(
        "--projection-time",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Look-ahead used to project track positions for the warning "
        "zone (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for DynamicPoints/StaticPoints/Targets/TargetsIndex "
        "files (default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live top view with both point clouds, tracked objects "
        "and the occupancy zones (requires: pip install urad-mmwave[gui])",
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

    # The Area Scanner firmware extends the out-of-box frame header with a
    # numStaticDetectedObj word; enforce it regardless of the profile.
    config.packet.header_format = HEADER_FORMAT

    zones = ZoneConfig(
        critical=tuple(args.critical_zone),
        warning=tuple(args.warning_zone),
        projection_time=args.projection_time,
    )
    output_dir = Path(args.output_dir)
    start_time = time()
    frame_count = 0

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            writers = {}
            if not args.no_save:
                for name in (
                    "DynamicPoints",
                    "StaticPoints",
                    "Targets",
                    "TargetsIndex",
                ):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def handle_frame(frame: AreaScannerFrame) -> None:
                status = classify_zones(frame, zones)
                print(
                    f"dynamic: {len(frame.dynamic_points)}  "
                    f"static: {len(frame.static_points)}  "
                    f"tracks: {len(frame.targets)}  "
                    f"zone: {status.label}"
                )

                if writers:
                    writers["DynamicPoints"].write_row(
                        [v for p in frame.dynamic_points for v in p], frame.timestamp
                    )
                    writers["StaticPoints"].write_row(
                        [v for p in frame.static_points for v in p], frame.timestamp
                    )
                    writers["Targets"].write_row(
                        [
                            v
                            for t in frame.targets
                            for v in (t.tid, *t.position, *t.velocity, *t.acceleration)
                        ],
                        frame.timestamp,
                    )
                    writers["TargetsIndex"].write_row(
                        list(frame.target_index), frame.timestamp
                    )

            def frames():
                for fields, payload, timestamp in session.packets():
                    yield parse_frame(
                        payload, timestamp, num_tlvs=fields[_HEADER_NUM_TLVS_INDEX]
                    )

            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration and --max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.area_scanner_viewer import run_viewer

                run_viewer(config, frames(), zones, on_frame=handle_frame)
            else:
                for frame in frames():
                    handle_frame(frame)
                    frame_count += 1
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

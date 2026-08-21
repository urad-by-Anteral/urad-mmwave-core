"""Automated Doors and Gates application (uRAD Industrial).

Requires the TI Automated Doors firmware (``automated_doors_68xx_demo_aop.bin``,
distributed in the uRAD Industrial repository). The firmware tracks people
and moving objects with the group tracker (GTRACK) and detects *newly added*
static objects close to the sensor (e.g. a cart left in the doorway) with a
range-angle heatmap subtraction algorithm, so a door controller can open
only for people actually approaching the door and refuse to close while the
doorway is obstructed.

The packet framing matches the out-of-box demo except for the frame header,
which appends a tenth word with the number of static detected objects
(44 bytes total, :data:`HEADER_FORMAT`). The TLV set is:

    1  — dynamic point cloud (range, azimuth, elevation in spherical
         coordinates, doppler)
    2  — range profile (uint16 per range bin, optional)
    3  — noise profile (uint16 per range bin, optional)
    6  — data path statistics (optional)
    7  — dynamic point cloud side info (snr, noise)
    8  — static point cloud (x, y, z cartesian, doppler)
    9  — static point cloud side info (snr, noise)
    10 — tracked object list (tid, position/velocity/acceleration)
    11 — point-to-track association (track id per dynamic point)

The open/close decision is not part of the UART stream: on the device it
drives a GPIO (``MmwDemo_setDoorState``, GPIO2 on the EVM) and the TI
MATLAB visualizer recomputes it host-side. :class:`DoorStateMachine`
reimplements that exact logic — a track triggers the door when it is inside
the approach zone and, at its current velocity, would reach the door plane
within ``open_time`` seconds; the door then stays open for ``hold_frames``
frames after the last activation. A static obstruction is flagged when any
newly added static point lies within ``obstruction_range`` of the sensor
(the state that turns the TI visualizer's door yellow).

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.automated_doors import (
        HEADER_FORMAT, DoorStateMachine, parse_frame,
    )

    config = load_config("config_radar.json")
    config.packet.header_format = HEADER_FORMAT
    door = DoorStateMachine()
    with RadarSession(config) as session:
        for fields, payload, timestamp in session.packets():
            frame = door.update(parse_frame(payload, timestamp, num_tlvs=fields[6]))
            print("open" if frame.door_open else "closed")
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
)
from urad_mmwave.config import load_config
from urad_mmwave.radar import RadarSession

log = logging.getLogger(__name__)

TLV_DYNAMIC_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_STATS = 6
TLV_DYNAMIC_SIDE_INFO = 7
TLV_STATIC_POINTS = 8
TLV_STATIC_SIDE_INFO = 9
TLV_TRACK_LIST = 10
TLV_TARGET_INDEX = 11

# Automated Doors frame header: the out-of-box header plus a trailing
# numStaticDetectedObj word (44 bytes). Field order after the sync word:
# version, totalPacketLen, platform, frameNumber, timeCpuCycles,
# numDetectedObj, numTLVs, subFrameNumber, numStaticDetectedObj.
HEADER_FORMAT = "<Q9I"
_HEADER_NUM_TLVS_INDEX = 6  # position of numTLVs in the fields after the sync

_DYNAMIC_POINT_STRUCT = struct.Struct("<4f")  # range, azimuth, elevation, doppler
_STATIC_POINT_STRUCT = struct.Struct("<4f")  # x, y, z, doppler
_SIDE_INFO_STRUCT = struct.Struct("<2h")  # snr, noise (int16, 0.1 dB steps)
# tid, posX, posY, velX, velY, accX, accY, posZ, velZ, accZ
_TARGET_STRUCT = struct.Struct("<I9f")
_STATS_STRUCT = struct.Struct("<6I")

# Firmware defaults of MmwDemo_setDoorState (mss_main.c): the approach zone
# is |x| < 2.0 m, 0 < y < 3.5 m; a track activates the door when it would
# reach y = 0 within 3 s (TIME_THRESHOLD) and the door is held open for 5
# frames (FDELAY) after the last activation. The 1.5 m obstruction radius
# comes from the TI visualizer (static objects within 1.5 m of the sensor
# flag an obstruction).
DOOR_HALF_WIDTH_M = 2.0
DOOR_DEPTH_M = 3.5
DOOR_OPEN_TIME_S = 3.0
DOOR_HOLD_FRAMES = 5
OBSTRUCTION_RANGE_M = 1.5


@dataclass(frozen=True)
class DoorTarget:
    """One tracked person/object from the track list TLV (type 10)."""

    tid: int
    position: tuple[float, float, float]  # x, y, z in meters
    velocity: tuple[float, float, float]  # m/s
    acceleration: tuple[float, float, float]  # m/s^2


@dataclass(frozen=True)
class DoorConfig:
    """Door trigger parameters (defaults match the TI firmware/visualizer).

    All coordinates are in the sensor frame — the same frame the firmware
    GPIO logic uses (no tilt rotation is applied).
    """

    half_width: float = DOOR_HALF_WIDTH_M  # approach zone: |x| < half_width (m)
    depth: float = DOOR_DEPTH_M  # approach zone: 0 < y < depth (m)
    open_time: float = DOOR_OPEN_TIME_S  # open when the door is < open_time s away
    hold_frames: int = DOOR_HOLD_FRAMES  # frames the door stays open after a trigger
    obstruction_range: float = OBSTRUCTION_RANGE_M  # static obstruction radius (m)


@dataclass(frozen=True)
class DataPathStats:
    """Data path timing statistics (TLV type 6), all values from the device."""

    inter_frame_processing_time_us: int
    transmit_output_time_us: int
    inter_frame_processing_margin_us: int
    inter_chirp_processing_margin_us: int
    active_frame_cpu_load_pct: int
    inter_frame_cpu_load_pct: int


@dataclass
class AutomatedDoorsFrame:
    """One decoded automated doors frame.

    Attributes:
        dynamic_points: Array of shape (N, 6) with columns range (m),
            azimuth (deg), elevation (deg), doppler (m/s), snr, noise
            (0.1 dB steps).
        static_points: Array of shape (M, 6) with columns x, y, z (m),
            doppler (m/s), snr, noise (0.1 dB steps). Only *newly added*
            static objects (after the heatmap calibration) appear here.
        targets: Tracked people/objects.
        target_index: Track id per dynamic point.
        range_profile: uint16 per range bin, if the firmware sends TLV 2.
        noise_profile: uint16 per range bin, if the firmware sends TLV 3.
        stats: Data path statistics, if the firmware sends TLV 6.
        door_open: Door trigger decision — True while a track approaches
            the door (or during the hold window). Set by
            :meth:`DoorStateMachine.update`; integrators should act on this.
        obstructed: True when a newly added static object sits within the
            obstruction radius of the sensor (door should not close).
        activated_tids: Track ids that triggered the door this frame.
        timestamp: Host epoch time when the frame was received.
    """

    dynamic_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    static_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    targets: list[DoorTarget] = field(default_factory=list)
    target_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    range_profile: np.ndarray | None = None
    noise_profile: np.ndarray | None = None
    stats: DataPathStats | None = None
    door_open: bool = False
    obstructed: bool = False
    activated_tids: tuple[int, ...] = ()
    timestamp: float = 0.0

    @property
    def door_label(self) -> str:
        """Human-readable door state for console output and the GUI."""
        if self.door_open:
            return "OPEN (obstructed)" if self.obstructed else "OPEN"
        return "OBSTRUCTED" if self.obstructed else "CLOSED"


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


def _parse_targets(body: bytes) -> list[DoorTarget]:
    targets = []
    for j in range(len(body) // _TARGET_STRUCT.size):
        tid, px, py, vx, vy, ax, ay, pz, vz, az = _TARGET_STRUCT.unpack_from(
            body, j * _TARGET_STRUCT.size
        )
        targets.append(
            DoorTarget(
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
) -> AutomatedDoorsFrame:
    """Decode the TLV payload of one automated doors packet.

    Args:
        payload: Packet bytes after the 44-byte frame header.
        timestamp: Host epoch time to attach to the frame.
        num_tlvs: TLV count from the frame header. Strongly recommended:
            this firmware pads packets with *uninitialized* bytes, so only
            the TLV count tells the padding apart from a further TLV.
            Without it the payload is scanned until an implausible TLV
            header is found.
    """
    frame = AutomatedDoorsFrame(timestamp=timestamp)
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
            break  # end-of-frame alignment padding
        if tlv_type > MAX_TLV_TYPE or tlv_length > MAX_TLV_LENGTH:
            if remaining is None:
                log.debug(
                    "Implausible TLV (type=%d, length=%d); assuming padding",
                    tlv_type,
                    tlv_length,
                )
            else:
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
        elif tlv_type == TLV_RANGE_PROFILE:
            frame.range_profile = np.frombuffer(body, dtype=np.uint16).astype(int)
        elif tlv_type == TLV_NOISE_PROFILE:
            frame.noise_profile = np.frombuffer(body, dtype=np.uint16).astype(int)
        elif tlv_type == TLV_STATS:
            if len(body) >= _STATS_STRUCT.size:
                frame.stats = DataPathStats(*_STATS_STRUCT.unpack_from(body, 0))
            else:
                log.warning("Stats TLV too short (%d bytes)", len(body))
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length
        if remaining is not None:
            remaining -= 1

    frame.dynamic_points = _with_side_info(dynamic, dynamic_side)
    frame.static_points = _with_side_info(static, static_side)
    return frame


def door_activates(target: DoorTarget, door: DoorConfig) -> bool:
    """True when ``target`` triggers the door opening this frame.

    Faithful reimplementation of the firmware's ``MmwDemo_setDoorState``:
    the track must be inside the approach zone (|x| < half_width,
    0 < y < depth) and moving towards the door plane (y = 0) fast enough to
    reach it within ``open_time`` seconds. A track walking away or across
    never activates.
    """
    x, y = target.position[0], target.position[1]
    vy = target.velocity[1]  # negative = moving towards the sensor
    if abs(x) >= door.half_width or not 0.0 < y < door.depth:
        return False
    if vy >= 0.0:
        return False  # static or moving away
    return y / vy > -door.open_time  # time to reach y=0 (negative when approaching)


def find_obstruction(frame: AutomatedDoorsFrame, door: DoorConfig) -> bool:
    """True when a newly added static object sits within the obstruction radius.

    The TI visualizer flags a static obstruction (yellow door) when a static
    point lies within 1.5 m of the sensor; the distance is the euclidean
    norm of the cartesian point.
    """
    if not len(frame.static_points):
        return False
    distances = np.linalg.norm(frame.static_points[:, :3], axis=1)
    return bool(np.any(distances <= door.obstruction_range))


class DoorStateMachine:
    """Stateful door open/close decision across frames.

    Mirrors the firmware GPIO behavior: activations re-arm a hold counter of
    ``door.hold_frames`` frames, so the door stays open briefly after the
    last person stopped triggering it (debounce against tracker flicker).
    """

    def __init__(self, door: DoorConfig | None = None):
        self.door = door or DoorConfig()
        self._hold = 0

    def update(self, frame: AutomatedDoorsFrame) -> AutomatedDoorsFrame:
        """Fill the door decision fields of ``frame`` and return it."""
        frame.activated_tids = tuple(
            target.tid for target in frame.targets if door_activates(target, self.door)
        )
        if frame.activated_tids:
            self._hold = self.door.hold_frames
            frame.door_open = True
        elif self._hold > 0:
            self._hold -= 1
            frame.door_open = True
        else:
            frame.door_open = False
        frame.obstructed = find_obstruction(frame, self.door)
        return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-automated-doors",
        description="Automated door/gate triggering with a uRAD Industrial "
        "radar running the Automated Doors firmware: opens only for people "
        "approaching the door and flags static obstructions in the doorway.",
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
        "--door-halfwidth",
        type=float,
        default=DOOR_HALF_WIDTH_M,
        metavar="METERS",
        help="Approach zone half width: tracks with |x| beyond this never "
        f"trigger the door (default: {DOOR_HALF_WIDTH_M})",
    )
    parser.add_argument(
        "--door-depth",
        type=float,
        default=DOOR_DEPTH_M,
        metavar="METERS",
        help="Approach zone depth: tracks beyond this y distance never "
        f"trigger the door (default: {DOOR_DEPTH_M})",
    )
    parser.add_argument(
        "--open-time",
        type=float,
        default=DOOR_OPEN_TIME_S,
        metavar="SECONDS",
        help="Open the door when an approaching track would reach it within "
        f"this many seconds (default: {DOOR_OPEN_TIME_S})",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=DOOR_HOLD_FRAMES,
        metavar="N",
        help="Keep the door open for this many frames after the last "
        f"activation (default: {DOOR_HOLD_FRAMES})",
    )
    parser.add_argument(
        "--obstruction-range",
        type=float,
        default=OBSTRUCTION_RANGE_M,
        metavar="METERS",
        help="Flag a static obstruction when a newly added static object is "
        f"within this distance of the sensor (default: {OBSTRUCTION_RANGE_M})",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for DynamicPoints/StaticPoints/Targets/TargetsIndex/"
        "DoorState files (default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live top view with the point clouds, tracked objects, "
        "the approach zone and a DOOR OPEN/CLOSED indicator "
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

    # The Automated Doors firmware extends the out-of-box frame header with
    # a numStaticDetectedObj word; enforce it regardless of the profile.
    config.packet.header_format = HEADER_FORMAT

    door = DoorConfig(
        half_width=args.door_halfwidth,
        depth=args.door_depth,
        open_time=args.open_time,
        hold_frames=args.hold_frames,
        obstruction_range=args.obstruction_range,
    )
    state_machine = DoorStateMachine(door)
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
                    "DoorState",
                ):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def handle_frame(frame: AutomatedDoorsFrame) -> None:
                print(
                    f"door: {frame.door_label:<17s}  "
                    f"tracks: {len(frame.targets)}  "
                    f"dynamic: {len(frame.dynamic_points)}  "
                    f"static: {len(frame.static_points)}"
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
                    writers["DoorState"].write_row(
                        [
                            int(frame.door_open),
                            int(frame.obstructed),
                            len(frame.activated_tids),
                        ],
                        frame.timestamp,
                    )

            def frames():
                for fields, payload, timestamp in session.packets():
                    yield state_machine.update(
                        parse_frame(
                            payload, timestamp, num_tlvs=fields[_HEADER_NUM_TLVS_INDEX]
                        )
                    )

            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration and --max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.automated_doors_viewer import run_viewer

                run_viewer(config, frames(), door, on_frame=handle_frame)
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

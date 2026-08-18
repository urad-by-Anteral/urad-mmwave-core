"""3D People Counting application (uRAD Industrial).

Requires the TI 3D People Counting firmware (standard or overhead variant,
distributed as release assets in the uRAD Industrial repository). Both
variants share this code — the difference is the firmware binary and the
chirp configuration file used.

The firmware shares the out-of-box packet framing (same sync word and
header) but uses its own TLV set:

    1010 — target list (tracked objects with position/velocity/acceleration)
    1011 — target index (track id per point of the previous frame)
    1012 — target height (estimated max/min Z per track)
    1020 — compressed spherical point cloud
    1021 — presence indication

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.people_counting import parse_frame

    with RadarSession(load_config("config_radar.json")) as session:
        for fields, payload, timestamp in session.packets():
            frame = parse_frame(payload, timestamp)
            print(len(frame.targets), "people tracked")
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
from urad_mmwave.config import load_config
from urad_mmwave.radar import RadarSession

log = logging.getLogger(__name__)

TLV_TARGET_LIST = 1010
TLV_TARGET_INDEX = 1011
TLV_TARGET_HEIGHT = 1012
TLV_POINT_CLOUD = 1020
TLV_PRESENCE = 1021

MAX_TLV_TYPE = 2000
MAX_TLV_LENGTH = 10000

_TLV_HEADER = struct.Struct("<2I")
_POINT_UNIT = struct.Struct("<5f")  # elevation, azimuth, doppler, range, snr units
# elevation, azimuth (int8), doppler (int16), range, snr (uint16).
# The legacy scripts decoded doppler as unsigned ('H'); the TI implementation
# guide defines it as signed, which this parser follows.
_POINT_STRUCT = struct.Struct("<2bh2H")
_TARGET_STRUCT = struct.Struct("<I9f16f2f")  # tid, pos/vel/acc, error cov., g, conf
_HEIGHT_STRUCT = struct.Struct("<B3x2f")  # tid, (padding), maxZ, minZ
_PRESENCE_STRUCT = struct.Struct("<I")


@dataclass(frozen=True)
class Target:
    """One tracked person from the target list TLV."""

    tid: int
    position: tuple[float, float, float]  # x, y, z in meters
    velocity: tuple[float, float, float]  # m/s
    acceleration: tuple[float, float, float]  # m/s^2
    confidence: float


@dataclass
class PeopleCountingFrame:
    """One decoded people counting frame.

    Attributes:
        points: Array of shape (N, 5) with columns range (m),
            azimuth (deg), elevation (deg), doppler (m/s), snr.
        targets: Tracked people.
        target_index: Track id per point of the *previous* frame's cloud.
        heights: Array of shape (M, 3) with columns tid, maxZ, minZ.
        presence: Presence indicator, if the firmware reports it.
        timestamp: Host epoch time when the frame was received.
    """

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    targets: list[Target] = field(default_factory=list)
    target_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    heights: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    presence: int | None = None
    timestamp: float = 0.0


def _parse_point_cloud(body: bytes) -> np.ndarray:
    if len(body) < _POINT_UNIT.size:
        log.warning("Point cloud TLV too short (%d bytes)", len(body))
        return np.zeros((0, 5))
    elev_unit, azim_unit, doppler_unit, range_unit, snr_unit = _POINT_UNIT.unpack_from(
        body, 0
    )
    count = (len(body) - _POINT_UNIT.size) // _POINT_STRUCT.size
    points = np.zeros((count, 5))
    for j in range(count):
        elevation, azimuth, doppler, range_, snr = _POINT_STRUCT.unpack_from(
            body, _POINT_UNIT.size + j * _POINT_STRUCT.size
        )
        points[j] = (
            range_ * range_unit,
            azimuth * azim_unit * 180.0 / np.pi,
            elevation * elev_unit * 180.0 / np.pi,
            doppler * doppler_unit,
            snr * snr_unit,
        )
    return points


def _parse_targets(body: bytes) -> list[Target]:
    targets = []
    for j in range(len(body) // _TARGET_STRUCT.size):
        fields = _TARGET_STRUCT.unpack_from(body, j * _TARGET_STRUCT.size)
        targets.append(
            Target(
                tid=fields[0],
                position=fields[1:4],
                velocity=fields[4:7],
                acceleration=fields[7:10],
                confidence=fields[-1],
            )
        )
    return targets


def _parse_heights(body: bytes) -> np.ndarray:
    count = len(body) // _HEIGHT_STRUCT.size
    heights = np.zeros((count, 3))
    for j in range(count):
        heights[j] = _HEIGHT_STRUCT.unpack_from(body, j * _HEIGHT_STRUCT.size)
    return heights


def parse_frame(payload: bytes, timestamp: float = 0.0) -> PeopleCountingFrame:
    """Decode the TLV payload of one people counting packet."""
    frame = PeopleCountingFrame(timestamp=timestamp)
    cursor = 0

    while cursor + _TLV_HEADER.size <= len(payload):
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

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
            frame.points = _parse_point_cloud(body)
        elif tlv_type == TLV_TARGET_LIST:
            frame.targets = _parse_targets(body)
        elif tlv_type == TLV_TARGET_INDEX:
            frame.target_index = np.frombuffer(body, dtype=np.uint8).astype(int)
        elif tlv_type == TLV_TARGET_HEIGHT:
            frame.heights = _parse_heights(body)
        elif tlv_type == TLV_PRESENCE:
            if len(body) >= _PRESENCE_STRUCT.size:
                frame.presence = _PRESENCE_STRUCT.unpack_from(body, 0)[0]
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length

    return frame


class _AppendWriter:
    """Append-only text writer in the legacy uRAD output format."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Kept open for the whole session (closed via close/__exit__).
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def write_row(self, values: list[float], timestamp: float) -> None:
        if not values:
            return
        parts = " ".join(
            str(int(v)) if float(v).is_integer() else f"{v:.3f}" for v in values
        )
        self._file.write(f"{parts} {timestamp:.3f}\n")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-people-counting",
        description="3D people counting with a uRAD Industrial radar running "
        "the People Counting firmware (standard or overhead).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the JSON configuration file (serial ports and chirp path)",
    )
    parser.add_argument("--control-port", help="Override the control serial port")
    parser.add_argument("--data-port", help="Override the data serial port")
    parser.add_argument("--chirp", help="Override the chirp configuration file path")
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for PointCloud/Targets/TargetsIndex/TargetsHeight files "
        "(default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl+C)",
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
    if args.chirp:
        config.chirp_config_path = args.chirp

    output_dir = Path(args.output_dir)
    start_time = time()

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            writers = {}
            if not args.no_save:
                for name in ("PointCloud", "Targets", "TargetsIndex", "TargetsHeight"):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            for _fields, payload, timestamp in session.packets():
                frame = parse_frame(payload, timestamp)
                print(
                    f"targets: {len(frame.targets)}  points: {len(frame.points)}"
                    + (
                        f"  presence: {frame.presence}"
                        if frame.presence is not None
                        else ""
                    )
                )

                if writers:
                    writers["PointCloud"].write_row(
                        [v for p in frame.points for v in p], timestamp
                    )
                    writers["Targets"].write_row(
                        [
                            v
                            for t in frame.targets
                            for v in (t.tid, *t.position, *t.velocity, *t.acceleration)
                        ],
                        timestamp,
                    )
                    writers["TargetsIndex"].write_row(
                        list(frame.target_index), timestamp
                    )
                    writers["TargetsHeight"].write_row(
                        [v for h in frame.heights for v in h], timestamp
                    )

                if args.duration is not None and time() - start_time >= args.duration:
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

"""Medium Range Radar (MRR) application (uRAD Automotive).

Requires the TI Medium Range Radar firmware (``xwr18xx_mrr_demo.bin``,
distributed with the uRAD Automotive repository). The firmware alternates
two subframes with different chirp designs:

- **MRR** subframe (subframe number 0): long range detection and tracking
  up to ~120 m with velocity disambiguation (max-velocity-enhancement
  processing path). Emits detected points and tracked objects.
- **USRR** subframe (subframe number 1): ultra short range, high resolution
  point cloud up to ~30 m (point cloud processing path). Emits detected
  points, DBSCAN clusters and the parking assist range profile.

Unlike other TI demos, the MRR firmware uses a **compiled-in RF
configuration** and starts streaming on boot: it accepts no CLI commands
at all (the CLI is compiled out), so there is no chirp ``.cfg`` file to
send and no ``sensorStop``. The client only reads the data UART, which
runs at 921600 baud (the auxiliary/data COM port of the board).

The packet framing is the standard mmWave demo one (same sync word and
40-byte header, with the subframe number as the last header field), but
the TLV set is specific to this demo:

    1 — detected points (descriptor + speed/peak/x/y/z per point)
    2 — clusters (descriptor + center/size per cluster, USRR only)
    3 — tracked objects (descriptor + position/velocity/size, MRR only)
    4 — parking assist (descriptor + nearest-obstruction range per
        azimuth bin, USRR only)

Every TLV starts with a descriptor giving the element count and the Q
format of the fixed-point values (2^-q meters or meters/second).

Typical usage:

    from urad_mmwave.apps.medium_range_radar import stream

    for frame in stream("COM7"):
        print(frame.subframe_name, len(frame.points), "points")
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import numpy as np

from urad_mmwave import __version__
from urad_mmwave.config import DEFAULT_HEADER_FORMAT, DEFAULT_SYNC_PATTERN, SerialConfig
from urad_mmwave.parser import iter_packets
from urad_mmwave.radar import _open_port, gpio_reset

log = logging.getLogger(__name__)

TLV_DETECTED_POINTS = 1
TLV_CLUSTERS = 2
TLV_TRACKED_OBJECTS = 3
TLV_PARKING_ASSIST = 4

# Header subframe number == the firmware processing path of the packet.
SUBFRAME_MRR = 0  # MAX_VEL_ENH_PROCESSING (long range + tracking)
SUBFRAME_USRR = 1  # POINT_CLOUD_PROCESSING (short range + clusters/parking)

DATA_BAUDRATE = 921600  # the firmware fixes the data UART at 115200 * 8

MAX_TLV_TYPE = 20
MAX_TLV_LENGTH = 10000

# The MSS pads each packet to a 32-byte multiple with 0x0F bytes; reading
# the padding as a TLV header yields this word.
PADDING_WORD = 0x0F0F0F0F

_TLV_HEADER = struct.Struct("<2I")
_DESCRIPTOR = struct.Struct("<2H")  # numDetectedObj, xyzQFormat
_POINT_STRUCT = struct.Struct("<hH3h")  # speed, peakVal, x, y, z
_CLUSTER_STRUCT = struct.Struct("<4h")  # xCenter, yCenter, xSize, ySize
_TRACKER_STRUCT = struct.Struct("<6h")  # x, y, xd, yd, xSize, ySize
_PARKING_BIN = struct.Struct("<H")

_DEFAULT_XYZ_QFORMAT = 7  # the firmware hardcodes 7 fractional bits


@dataclass(frozen=True)
class TrackedObject:
    """One tracked object from the MRR subframe (EKF tracker output)."""

    position: tuple[float, float]  # x, y in meters
    velocity: tuple[float, float]  # vx, vy in m/s
    size: tuple[float, float]  # cluster extents (x, y) in meters


@dataclass
class MrrFrame:
    """One decoded Medium Range Radar frame (one subframe of the sensor).

    Attributes:
        subframe: ``SUBFRAME_MRR`` (0) or ``SUBFRAME_USRR`` (1).
        points: Array of shape (N, 5) with columns x (m), y (m), z (m),
            doppler (m/s, negative = approaching) and peak value (raw
            linear units).
        clusters: Array of shape (M, 4) with columns xCenter, yCenter,
            xSize, ySize in meters (sizes are half-extents; USRR only).
        trackers: Tracked objects (MRR only).
        parking_assist: Array of nearest-obstruction ranges in meters,
            one per azimuth bin (USRR only; see
            :func:`parking_assist_to_xy` for the bin layout).
        frame_number: Frame counter from the packet header.
        timestamp: Host epoch time when the frame was received.
    """

    subframe: int = SUBFRAME_MRR
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    clusters: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    trackers: list[TrackedObject] = field(default_factory=list)
    parking_assist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frame_number: int = 0
    timestamp: float = 0.0

    @property
    def subframe_name(self) -> str:
        """Human-readable subframe name (``"MRR"`` or ``"USRR"``)."""
        return "USRR" if self.subframe == SUBFRAME_USRR else "MRR"


def _q_scale(q_format: int) -> float:
    """Scale factor for a Q-format descriptor value, with a sane fallback."""
    if not 0 < q_format <= 15:
        log.debug("Implausible xyzQFormat %d; assuming 7", q_format)
        q_format = _DEFAULT_XYZ_QFORMAT
    return 1.0 / (1 << q_format)


def _parse_points(body: bytes) -> np.ndarray:
    if len(body) < _DESCRIPTOR.size:
        log.warning("Detected points TLV too short (%d bytes)", len(body))
        return np.zeros((0, 5))
    declared, q_format = _DESCRIPTOR.unpack_from(body, 0)
    scale = _q_scale(q_format)
    count = min(declared, (len(body) - _DESCRIPTOR.size) // _POINT_STRUCT.size)
    points = np.zeros((count, 5))
    for j in range(count):
        speed, peak_val, x, y, z = _POINT_STRUCT.unpack_from(
            body, _DESCRIPTOR.size + j * _POINT_STRUCT.size
        )
        points[j] = (x * scale, y * scale, z * scale, speed * scale, peak_val)
    return points


def _parse_clusters(body: bytes) -> np.ndarray:
    if len(body) < _DESCRIPTOR.size:
        log.warning("Clusters TLV too short (%d bytes)", len(body))
        return np.zeros((0, 4))
    declared, q_format = _DESCRIPTOR.unpack_from(body, 0)
    scale = _q_scale(q_format)
    count = min(declared, (len(body) - _DESCRIPTOR.size) // _CLUSTER_STRUCT.size)
    clusters = np.zeros((count, 4))
    for j in range(count):
        clusters[j] = _CLUSTER_STRUCT.unpack_from(
            body, _DESCRIPTOR.size + j * _CLUSTER_STRUCT.size
        )
    return clusters * scale


def _parse_trackers(body: bytes) -> list[TrackedObject]:
    if len(body) < _DESCRIPTOR.size:
        log.warning("Tracked objects TLV too short (%d bytes)", len(body))
        return []
    declared, q_format = _DESCRIPTOR.unpack_from(body, 0)
    scale = _q_scale(q_format)
    count = min(declared, (len(body) - _DESCRIPTOR.size) // _TRACKER_STRUCT.size)
    trackers = []
    for j in range(count):
        x, y, xd, yd, x_size, y_size = _TRACKER_STRUCT.unpack_from(
            body, _DESCRIPTOR.size + j * _TRACKER_STRUCT.size
        )
        trackers.append(
            TrackedObject(
                position=(x * scale, y * scale),
                velocity=(xd * scale, yd * scale),
                size=(x_size * scale, y_size * scale),
            )
        )
    return trackers


def _parse_parking_assist(body: bytes) -> np.ndarray:
    if len(body) < _DESCRIPTOR.size:
        log.warning("Parking assist TLV too short (%d bytes)", len(body))
        return np.zeros(0)
    declared, q_format = _DESCRIPTOR.unpack_from(body, 0)
    scale = _q_scale(q_format)
    count = min(declared, (len(body) - _DESCRIPTOR.size) // _PARKING_BIN.size)
    bins = np.frombuffer(
        body, dtype="<u2", count=count, offset=_DESCRIPTOR.size
    ).astype(float)
    return bins * scale


def parking_assist_to_xy(ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the parking assist bins to x/y meters, ordered left to right.

    The firmware quantizes sin(azimuth) over the full [-1, 1) span into N
    bins: bins ``0 .. N/2-1`` cover sin(azimuth) in [0, 1) (boresight to
    the right) and bins ``N/2 .. N-1`` cover [-1, 0) (left of boresight).
    Each bin holds the range of the nearest obstruction, clamped to the
    firmware's maximum parking range (20 m) when the bin is free.
    """
    count = len(ranges)
    if not count:
        return np.zeros(0), np.zeros(0)
    sin_azimuth = 2.0 * np.arange(count) / count
    sin_azimuth[sin_azimuth >= 1.0] -= 2.0
    order = np.argsort(sin_azimuth)
    sin_azimuth = sin_azimuth[order]
    ranges = np.asarray(ranges)[order]
    cos_azimuth = np.sqrt(1.0 - sin_azimuth**2)
    return ranges * sin_azimuth, ranges * cos_azimuth


def parse_frame(
    payload: bytes, subframe: int = SUBFRAME_MRR, timestamp: float = 0.0
) -> MrrFrame:
    """Decode the TLV payload of one Medium Range Radar packet.

    ``subframe`` is the last field of the packet header (0 = MRR subframe,
    1 = USRR subframe) and selects how the frame content is interpreted
    downstream; the TLVs themselves are parsed by type either way.
    """
    frame = MrrFrame(subframe=subframe, timestamp=timestamp)
    cursor = 0

    while cursor + _TLV_HEADER.size <= len(payload):
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

        if tlv_type == PADDING_WORD:
            break  # end-of-frame alignment padding (0x0F bytes)
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

        if tlv_type == TLV_DETECTED_POINTS:
            frame.points = _parse_points(body)
        elif tlv_type == TLV_CLUSTERS:
            frame.clusters = _parse_clusters(body)
        elif tlv_type == TLV_TRACKED_OBJECTS:
            frame.trackers = _parse_trackers(body)
        elif tlv_type == TLV_PARKING_ASSIST:
            frame.parking_assist = _parse_parking_assist(body)
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length

    return frame


def stream(
    data_port: str,
    baudrate: int = DATA_BAUDRATE,
    gpio_reset_pin: int | None = None,
    timeout: float = 0.3,
) -> Iterator[MrrFrame]:
    """Yield :class:`MrrFrame` continuously until the generator is closed.

    The MRR firmware configures itself and starts streaming on boot, so
    this only opens the data UART — no configuration commands are sent and
    the sensor keeps transmitting after the port is released (power-cycle
    or reset the board to stop it).
    """
    if gpio_reset_pin is not None:
        log.info("Resetting radar via GPIO pin %d", gpio_reset_pin)
        gpio_reset(gpio_reset_pin)

    data_cfg = SerialConfig(port=data_port, baudrate=baudrate, timeout=timeout)
    log.info("Opening data port %s at %d baud", data_cfg.port, data_cfg.baudrate)
    port = _open_port(data_cfg)
    try:
        port.reset_input_buffer()
        for fields, payload, timestamp in iter_packets(
            port, DEFAULT_SYNC_PATTERN, DEFAULT_HEADER_FORMAT, max_empty_reads=100
        ):
            # Header fields: version, totalPacketLen, platform, frameNumber,
            # timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber.
            frame = parse_frame(payload, subframe=fields[7], timestamp=timestamp)
            frame.frame_number = fields[3]
            yield frame
    finally:
        if port.is_open:
            port.close()


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
        prog="urad-mrr",
        description="Medium Range Radar (MRR/USRR) with a uRAD Automotive "
        "radar running the TI Medium Range Radar firmware. The firmware "
        "uses a compiled-in configuration and streams on boot, so no chirp "
        "file is needed — only the data serial port.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Optional JSON configuration file (data serial port section); "
        "the chirp configuration path is ignored — this firmware has none",
    )
    parser.add_argument(
        "--data-port",
        help="Data serial port (e.g. COM7); required unless --config is given",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DATA_BAUDRATE,
        help=f"Data UART baud rate (default: {DATA_BAUDRATE}, fixed by the "
        "firmware)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for PointCloud/Clusters/Trackers/ParkingAssist files "
        "(default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live top view with MRR/USRR points, clusters, tracked "
        "objects and the parking assist border "
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
        help="Stop after N frames (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--gpio-reset-pin",
        type=int,
        help="BCM pin to reset the chip before reading (Raspberry Pi)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    data_port = args.data_port
    baudrate = args.baudrate
    if args.config:
        from urad_mmwave.config import load_config

        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s", exc)
            return 1
        if data_port is None:
            data_port = config.data_serial.port
        if args.baudrate == DATA_BAUDRATE:
            baudrate = config.data_serial.baudrate
    if data_port is None:
        parser.error("--data-port is required (or provide it via --config)")

    output_dir = Path(args.output_dir)
    start_time = time()
    frame_count = 0

    try:
        with ExitStack() as stack:
            writers = {}
            if not args.no_save:
                for name in ("PointCloud", "Clusters", "Trackers", "ParkingAssist"):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def handle_frame(frame: MrrFrame) -> None:
                if frame.subframe == SUBFRAME_USRR:
                    print(
                        f"USRR  points: {len(frame.points):3d}  "
                        f"clusters: {len(frame.clusters)}"
                    )
                else:
                    print(
                        f"MRR   points: {len(frame.points):3d}  "
                        f"trackers: {len(frame.trackers)}"
                    )

                if writers:
                    writers["PointCloud"].write_row(
                        [frame.subframe] + [v for p in frame.points for v in p],
                        frame.timestamp,
                    )
                    writers["Clusters"].write_row(
                        [v for c in frame.clusters for v in c], frame.timestamp
                    )
                    writers["Trackers"].write_row(
                        [
                            v
                            for t in frame.trackers
                            for v in (*t.position, *t.velocity, *t.size)
                        ],
                        frame.timestamp,
                    )
                    writers["ParkingAssist"].write_row(
                        list(frame.parking_assist), frame.timestamp
                    )

            frames = stream(
                data_port,
                baudrate=baudrate,
                gpio_reset_pin=args.gpio_reset_pin,
            )
            stack.callback(frames.close)  # release the port on any exit path

            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration/--max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.medium_range_radar_viewer import run_viewer

                run_viewer(frames, on_frame=handle_frame)
            else:
                for frame in frames:
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
        log.info("Interrupted by user; closing port (the sensor keeps streaming)")
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of a traceback
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

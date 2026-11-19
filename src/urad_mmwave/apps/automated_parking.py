"""Automated Parking application (uRAD Automotive and uRAD Automotive HPA).

Client for the TI Radar Toolbox **Automated Parking** lab: the mmWave SDK
demo firmware configured with an *advanced frame* of two subframes, so one
radar covers two zones at once:

- **Subframe 0 — long range 2D** (0-50 m): coarse range resolution, X-Y
  point cloud to see vehicles and pedestrians approaching from afar.
- **Subframe 1 — short range 3D** (0-10 m): three TX in MIMO for elevation,
  so every point also has a height (z) — kerbs, posts and low obstacles
  close to the vehicle.

Each subframe arrives as its own packet in the standard mmWave SDK output
format (detected points, side info, range profile, statistics), with the
subframe number in the header, so the generic out-of-box parser decodes the
points; this module separates the two zones, adds the range profile (TLV 2)
and reports the nearest short range obstacle.

The firmware and the chirp configuration of each product live in the
``applications/automated_parking`` folder of the product repositories
(AWR1843AoP for the uRAD Automotive, AWR1843BOOST antenna layout for the
HPA).

Typical usage:

    urad-parking --config config_radar.json --gui
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
from urad_mmwave.config import PacketConfig, load_config
from urad_mmwave.parser import Frame, FrameHeader, parse_tlvs
from urad_mmwave.radar import RadarSession, read_chirp_config
from urad_mmwave.writer import PointCloudWriter

log = logging.getLogger(__name__)

SUBFRAME_LONG_RANGE = 0
"""Long range 2D subframe (0-50 m)."""
SUBFRAME_SHORT_RANGE = 1
"""Short range 3D subframe (0-10 m)."""
SUBFRAME_NAMES = {SUBFRAME_LONG_RANGE: "2D 50 m", SUBFRAME_SHORT_RANGE: "3D 10 m"}

TLV_RANGE_PROFILE = 2
_TLV_HEADER = struct.Struct("<2I")
_LOG2_TO_DB = 20 * np.log10(2)  # range profile: log2 magnitude in Q9


@dataclass
class ParkingFrame:
    """One subframe of the Automated Parking radar.

    Attributes:
        subframe: ``SUBFRAME_LONG_RANGE`` (0) or ``SUBFRAME_SHORT_RANGE`` (1).
        points: (n, 6) array x, y, z (m), v (m/s), snr, noise (0.1 dB units).
            In the long range subframe z is 0 (no elevation).
        range_profile_db: Range profile at zero Doppler in dB (relative),
            one value per range bin; empty if the frame did not carry it.
        frame_number: Frame counter from the packet header.
        timestamp: Host epoch time when the packet was received.
    """

    subframe: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    range_profile_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frame_number: int = 0
    timestamp: float = 0.0

    @property
    def name(self) -> str:
        return SUBFRAME_NAMES.get(self.subframe, f"subframe {self.subframe}")


def parse_range_profile(payload: bytes, num_tlvs: int) -> np.ndarray:
    """Range profile (TLV 2) of an mmWave SDK packet, in dB (relative).

    The firmware sends one uint16 per range bin: the sum over the virtual
    antennas of the log2 magnitude, in Q9. Returns an empty array when the
    packet has no range profile.
    """
    offset = 0
    for _ in range(num_tlvs):
        if offset + _TLV_HEADER.size > len(payload):
            break
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, offset)
        offset += _TLV_HEADER.size
        if offset + tlv_length > len(payload):
            break
        if tlv_type == TLV_RANGE_PROFILE:
            raw = np.frombuffer(
                payload, dtype="<u2", count=tlv_length // 2, offset=offset
            )
            return raw.astype(float) / 512.0 * _LOG2_TO_DB
        offset += tlv_length
    return np.zeros(0)


def range_bin_sizes(commands: list[str]) -> dict[int, float]:
    """Range covered by one range-profile bin (m) for each subframe.

    Follows ``subFrameCfg`` (first chirp of the subframe) → ``chirpCfg``
    (its profile) → ``profileCfg`` (slope, samples, sampling rate):
    ``c · fs / (2 · slope · N_fft)`` with N_fft the next power of two of the
    number of ADC samples. Returns an empty dict if the lines are missing.
    """
    profiles, chirps, subframes = {}, {}, {}
    for line in commands:
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "profileCfg":
                profiles[int(parts[1])] = (
                    float(parts[8]),
                    int(parts[10]),
                    float(parts[11]),
                )
            elif parts[0] == "chirpCfg":
                for idx in range(int(parts[1]), int(parts[2]) + 1):
                    chirps[idx] = int(parts[3])
            elif parts[0] == "subFrameCfg":
                subframes[int(parts[1])] = int(parts[3])
        except (IndexError, ValueError):
            log.debug("Unparsable line for range bins: %s", line)
    out = {}
    for sub, first_chirp in subframes.items():
        profile = profiles.get(chirps.get(first_chirp, -1))
        if profile is None:
            continue
        slope_mhz_us, samples, fs_ksps = profile
        n_fft = 1 << (samples - 1).bit_length()
        out[sub] = 299_792_458.0 * fs_ksps * 1e3 / (2 * slope_mhz_us * 1e12 * n_fft)
    return out


def nearest_obstacle(points: np.ndarray) -> float | None:
    """Horizontal distance (m) to the closest point, or None if no points."""
    if len(points) == 0:
        return None
    return float(np.min(np.hypot(points[:, 0], points[:, 1])))


def to_parking_frame(frame: Frame, payload: bytes | None = None) -> ParkingFrame:
    """Wrap a decoded out-of-box frame (plus its raw payload for the range
    profile) as a :class:`ParkingFrame`."""
    profile = (
        parse_range_profile(payload, frame.header.num_tlvs)
        if payload is not None
        else np.zeros(0)
    )
    return ParkingFrame(
        subframe=frame.header.subframe_number,
        points=frame.points,
        range_profile_db=profile,
        frame_number=frame.header.frame_number,
        timestamp=frame.timestamp,
    )


def parking_frames(
    packets: Iterator[tuple[tuple, bytes, float]], packet_config: PacketConfig
) -> Iterator[ParkingFrame]:
    """Decode raw packets (e.g. ``RadarSession.packets()``) into
    :class:`ParkingFrame` objects."""
    for fields, payload, timestamp in packets:
        header = FrameHeader(*fields)
        points, _ = parse_tlvs(payload, header, packet_config)
        frame = Frame(header=header, points=points, timestamp=timestamp)
        yield to_parking_frame(frame, payload)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-parking",
        description="Automated Parking with a uRAD Automotive or uRAD Automotive "
        "HPA: long range 2D (0-50 m) and short range 3D (0-10 m) point clouds "
        "from one radar.",
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
        "(e.g. /dev/serial0 on Raspberry Pi)",
    )
    parser.add_argument("--chirp", help="Override the chirp configuration file path")
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for LongRange.txt and ShortRange.txt (default: ./output)",
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
        help="Show the live views of both zones and the range profiles "
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
        help="Stop after N packets, counting both subframes (default: unlimited)",
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


def format_frame(frame: ParkingFrame) -> str:
    """Console line for one subframe."""
    line = f"{frame.name}  frame {frame.frame_number}: {len(frame.points)} points"
    if frame.subframe == SUBFRAME_SHORT_RANGE:
        nearest = nearest_obstacle(frame.points)
        if nearest is not None:
            line += f", nearest obstacle {nearest:.2f} m"
    return line


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

    output_dir = Path(args.output_dir)
    start_time = time()
    count = 0

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))
            writers = {}
            if not args.no_save:
                writers[SUBFRAME_LONG_RANGE] = stack.enter_context(
                    PointCloudWriter(output_dir / "LongRange.txt")
                )
                writers[SUBFRAME_SHORT_RANGE] = stack.enter_context(
                    PointCloudWriter(output_dir / "ShortRange.txt")
                )

            def handle(frame: ParkingFrame) -> None:
                print(format_frame(frame))
                if args.print_points:
                    for p in frame.points:
                        print(
                            f"  x: {p[0]:.3f} m, y: {p[1]:.3f} m, z: {p[2]:.3f} m, "
                            f"v: {p[3]:.3f} m/s, snr: {int(p[4])}, noise: {int(p[5])}"
                        )
                writer = writers.get(frame.subframe)
                if writer is not None:
                    writer.write(frame.points, frame.timestamp)

            frames = parking_frames(session.packets(), config.packet)
            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration and --max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.automated_parking_viewer import run_viewer

                bins = range_bin_sizes(read_chirp_config(config.chirp_config_path))
                run_viewer(frames, on_frame=handle, range_bin_m=bins)
            else:
                for frame in frames:
                    handle(frame)
                    count += 1
                    if args.max_frames is not None and count >= args.max_frames:
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

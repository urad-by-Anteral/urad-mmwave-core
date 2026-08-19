"""High Accuracy Level Sensing application (uRAD Automotive and Industrial).

Requires the uRAD Level Sensing firmware (distributed as a release asset in
the product repositories); it does not run on the out-of-box demo firmware.
The radar reports three high-accuracy range measurements per frame using a
fixed-point encoding. The chirp configuration is generated from the
measurement parameters instead of being read from a ``.cfg`` file, so the
same code serves both products: ``model="AWR"`` (Automotive, 77 GHz) and
``model="IWR"`` (Industrial, 60 GHz).

Typical usage:

    from urad_mmwave.apps.level_sensing import LevelSensingSettings, measure

    settings = LevelSensingSettings(model="IWR", maximum_distance=12)
    result = measure(settings, control_port="COM8", data_port="COM7")
    print(result.ranges)
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from dataclasses import dataclass
from time import sleep, time

import numpy as np

from urad_mmwave import __version__
from urad_mmwave.config import DEFAULT_SYNC_PATTERN, SerialConfig
from urad_mmwave.parser import StreamTimeoutError, iter_packets
from urad_mmwave.radar import _open_port, _send_command, gpio_reset

log = logging.getLogger(__name__)

HEADER_FORMAT = "<Q7I"  # sync + version, totalLen, platform, frame, cpu, objs, tlvs
TLV_HEADER_FORMAT = "<2I"
TLV_RANGES = 1

_TLV_HEADER = struct.Struct(TLV_HEADER_FORMAT)
_DESCRIPTOR_STRUCT = struct.Struct("<2H")  # numDetectedObj, xyzQFormat
# r1_low, r3_low, r2_low, r1, r2, r3. The low halves are unsigned: the
# legacy scripts decoded r3_low as signed (a leftover of the out-of-box
# point struct, where that slot is a signed doppler index), which made
# range 3 read 62.5 mm short whenever its low word was >= 0x8000.
_RANGES_STRUCT = struct.Struct("<3H3h")

_CHIRP_SLOPE_MAX = 31.23
_CHIRP_SLOPE_MIN = 1.87
_SLOPE_CONSTANT = 374.7405725
_SAMPLING_RATE_HZ = 20
_MIN_DISTANCE_M = 12
_MAX_DISTANCE_M = 150
_FIXED_POINT_SCALE = 2**20


@dataclass
class LevelSensingSettings:
    """Measurement parameters for the level sensing firmware."""

    model: str  # "AWR" (Automotive, 77 GHz) or "IWR" (Industrial, 60 GHz)
    maximum_distance: float = 12.0  # meters, clamped to [12, 150]
    range_min: float = 0.0  # lower limit of the range of interest
    range_max: float = 10.0  # upper limit of the range of interest
    offset: float = 0.0  # added to every measured range
    n_measurements: int = 5  # final result is the mean of N frames

    def __post_init__(self) -> None:
        if self.model not in ("AWR", "IWR"):
            raise ValueError(f"model must be 'AWR' or 'IWR', got {self.model!r}")


@dataclass(frozen=True)
class Measurement:
    """Averaged level sensing result (meters)."""

    ranges: tuple[float, float, float]
    n_frames: int
    timestamp: float


def chirp_slope(maximum_distance: float) -> float:
    """Chirp slope (MHz/us) for a maximum measurable distance in meters."""
    if maximum_distance < _MIN_DISTANCE_M:
        return _CHIRP_SLOPE_MAX
    if maximum_distance > _MAX_DISTANCE_M:
        return _CHIRP_SLOPE_MIN
    return _SLOPE_CONSTANT / maximum_distance


def build_commands(settings: LevelSensingSettings) -> list[str]:
    """Generate the radar configuration commands for the given settings."""
    range_min, range_max = settings.range_min, settings.range_max
    if range_min > range_max:
        range_min, range_max = range_max, range_min
    if range_min < 0:
        range_min = 0.0
    if range_max < 0:
        range_max = min(settings.maximum_distance, _MAX_DISTANCE_M)

    start_freq = 77 if settings.model == "AWR" else 60
    slope = chirp_slope(settings.maximum_distance)
    frame_period_ms = 1e3 / _SAMPLING_RATE_HZ

    return [
        "flushCfg",
        "dfeDataOutputMode 1",
        "channelCfg 1 1 0",
        "adcCfg 2 1",
        "adcbufCfg 0 1 1 1",
        f"profileCfg 0 {start_freq} 7 7 114.4 0 0 {slope:.4f} 1 512 5000 0 0 48",
        "chirpCfg 0 0 0 0 0 0 0 1",
        f"frameCfg 0 0 10 0 {frame_period_ms:.0f} 1 0",
        "lowPower 0 0",
        "guiMonitor 1 0 0 0 0 1",
        f"RangeLimitCfg 2 1 {range_min:.1f} {range_max:.1f}",
        "sensorStart",
    ]


def decode_ranges(
    payload: bytes, offset: float = 0.0
) -> tuple[float, float, float] | None:
    """Extract the three fixed-point ranges from one frame payload.

    Returns None if the payload does not carry a ranges TLV (type 1).
    """
    cursor = 0
    while cursor + _TLV_HEADER.size <= len(payload):
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

        if tlv_type == TLV_RANGES:
            needed = _DESCRIPTOR_STRUCT.size + _RANGES_STRUCT.size
            if cursor + needed > len(payload):
                log.warning("Truncated ranges TLV; skipping frame")
                return None
            r1_low, r3_low, r2_low, r1, r2, r3 = _RANGES_STRUCT.unpack_from(
                payload, cursor + _DESCRIPTOR_STRUCT.size
            )
            return (
                ((r1 << 16) + r1_low) / _FIXED_POINT_SCALE + offset,
                ((r2 << 16) + r2_low) / _FIXED_POINT_SCALE + offset,
                ((r3 << 16) + r3_low) / _FIXED_POINT_SCALE + offset,
            )

        cursor += tlv_length

    return None


def stream(
    settings: LevelSensingSettings,
    control_port: str,
    data_port: str | None = None,
    gpio_reset_pin: int | None = None,
    timeout: float = 1.0,
):
    """Yield ``(timestamp, (r1, r2, r3))`` continuously until closed.

    Configures the radar once and streams decoded range frames; the sensor
    is stopped and the ports released when the generator is closed (also on
    error). Used by the live viewer (``--gui``).
    """
    single_port = data_port is None or data_port == control_port
    control_cfg = SerialConfig(port=control_port, baudrate=115200, timeout=0.3)
    data_cfg = SerialConfig(
        port=control_port if single_port else data_port,
        baudrate=921600,
        timeout=timeout,
    )

    if gpio_reset_pin is not None:
        log.info("Resetting radar via GPIO pin %d", gpio_reset_pin)
        gpio_reset(gpio_reset_pin)

    commands = build_commands(settings)
    log.info(
        "Configuring level sensing on %s (%s, max %.0f m)",
        control_cfg.port,
        settings.model,
        settings.maximum_distance,
    )

    port = None
    try:
        with _open_port(control_cfg) as control:
            control.reset_input_buffer()
            for command in commands:
                response = _send_command(control, command)
                log.debug("%s -> %s", command, response or "<no response>")

        port = _open_port(data_cfg)
        port.reset_input_buffer()

        for _fields, payload, timestamp in iter_packets(
            port, DEFAULT_SYNC_PATTERN, HEADER_FORMAT, max_empty_reads=100
        ):
            ranges = decode_ranges(payload, settings.offset)
            if ranges is not None:
                yield timestamp, ranges
    finally:
        if port is not None and port.is_open:
            port.close()
        try:
            with _open_port(control_cfg) as control:
                _send_command(control, "sensorStop")
            log.info("Sensor stopped")
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask errors
            log.warning("Could not send sensorStop: %s", exc)


def measure(
    settings: LevelSensingSettings,
    control_port: str,
    data_port: str | None = None,
    gpio_reset_pin: int | None = None,
    timeout: float = 1.0,
) -> Measurement:
    """Run one averaged measurement cycle and stop the sensor.

    For single-UART setups (e.g. Raspberry Pi) pass only ``control_port``;
    the same physical port is reused for data at 921600 baud. The sensor is
    always stopped and the ports closed, even on error.

    Raises:
        TimeoutError: If not enough valid frames arrive.
    """
    single_port = data_port is None or data_port == control_port
    control_cfg = SerialConfig(port=control_port, baudrate=115200, timeout=0.3)
    data_cfg = SerialConfig(
        port=control_port if single_port else data_port,
        baudrate=921600,
        timeout=timeout,
    )

    if gpio_reset_pin is not None:
        log.info("Resetting radar via GPIO pin %d", gpio_reset_pin)
        gpio_reset(gpio_reset_pin)

    commands = build_commands(settings)
    log.info(
        "Configuring level sensing on %s (%s, max %.0f m)",
        control_cfg.port,
        settings.model,
        settings.maximum_distance,
    )

    collected: list[tuple[float, float, float]] = []
    port = None
    try:
        with _open_port(control_cfg) as control:
            control.reset_input_buffer()
            for command in commands:
                response = _send_command(control, command)
                log.debug("%s -> %s", command, response or "<no response>")

        port = _open_port(data_cfg)
        port.reset_input_buffer()

        max_frames = 5 * settings.n_measurements
        frames_seen = 0
        try:
            for _fields, payload, _ts in iter_packets(
                port, DEFAULT_SYNC_PATTERN, HEADER_FORMAT, max_empty_reads=30
            ):
                frames_seen += 1
                ranges = decode_ranges(payload, settings.offset)
                if ranges is not None:
                    collected.append(ranges)
                if (
                    len(collected) >= settings.n_measurements
                    or frames_seen >= max_frames
                ):
                    break
        except StreamTimeoutError as exc:
            log.warning("%s", exc)
    finally:
        if port is not None and port.is_open:
            port.close()
        try:
            with _open_port(control_cfg) as control:
                _send_command(control, "sensorStop")
            log.info("Sensor stopped")
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask errors
            log.warning("Could not send sensorStop: %s", exc)

    if not collected:
        raise TimeoutError("No valid level sensing frames received")

    means = np.mean(np.asarray(collected), axis=0)
    return Measurement(
        ranges=(float(means[0]), float(means[1]), float(means[2])),
        n_frames=len(collected),
        timestamp=time(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-level-sensing",
        description="High accuracy level sensing with a uRAD mmWave radar "
        "running the Level Sensing firmware.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["AWR", "IWR"],
        help="AWR = uRAD Automotive (77 GHz), IWR = uRAD Industrial (60 GHz)",
    )
    parser.add_argument(
        "--control-port", required=True, help="Control serial port (e.g. COM8)"
    )
    parser.add_argument(
        "--data-port",
        help="Data serial port; omit for single-UART setups (Raspberry Pi)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=12.0,
        help="Maximum measurable distance in meters, 12-150 (default: 12)",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        default=[0.0, 10.0],
        metavar=("MIN", "MAX"),
        help="Range of interest in meters (default: 0 10)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Offset added to every measured range (default: 0)",
    )
    parser.add_argument(
        "-n",
        "--measurements",
        type=int,
        default=5,
        help="Frames averaged per measurement (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        metavar="SECONDS",
        help="Repeat every N seconds (default: measure once and exit)",
    )
    parser.add_argument(
        "--gpio-reset-pin",
        type=int,
        help="BCM pin to reset the chip before measuring (Raspberry Pi)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the three ranges live over time "
        "(requires: pip install urad-mmwave[gui])",
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

    settings = LevelSensingSettings(
        model=args.model,
        maximum_distance=args.max_distance,
        range_min=args.range[0],
        range_max=args.range[1],
        offset=args.offset,
        n_measurements=args.measurements,
    )

    try:
        if args.gui:
            if args.interval is not None:
                log.warning(
                    "--interval is ignored in GUI mode; close the window to stop"
                )
            from urad_mmwave.apps.level_sensing_viewer import run_viewer

            samples = stream(
                settings,
                control_port=args.control_port,
                data_port=args.data_port,
                gpio_reset_pin=args.gpio_reset_pin,
            )
            try:
                run_viewer(samples)
            finally:
                samples.close()  # runs the generator cleanup (sensorStop)
            return 0

        while True:
            result = measure(
                settings,
                control_port=args.control_port,
                data_port=args.data_port,
                gpio_reset_pin=args.gpio_reset_pin,
            )
            print(
                f"{result.ranges[0]:.4f} {result.ranges[1]:.4f} "
                f"{result.ranges[2]:.4f}"
            )
            if args.interval is None:
                break
            sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of a traceback
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

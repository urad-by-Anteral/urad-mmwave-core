"""Command line interface for the uRAD mmWave SDK.

Configures the radar, streams frames and prints/saves the point cloud:

    urad-mmwave --config config_radar.json
    urad-mmwave --config config_radar.json --duration 10
    urad-mmwave --config config_radar.json --data-port COM7 --control-port COM8
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import ExitStack
from time import time

from urad_mmwave import __version__
from urad_mmwave.config import load_config
from urad_mmwave.parser import Frame
from urad_mmwave.radar import RadarSession
from urad_mmwave.writer import PointCloudWriter, TemperatureWriter

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-mmwave",
        description="Stream point cloud data from a uRAD mmWave radar "
        "running the out-of-box demo firmware.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./config/config_radar.json",
        help="Path to the JSON configuration file "
        "(default: ./config/config_radar.json)",
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
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live 2D point cloud viewer "
        "(requires: pip install urad-mmwave[gui])",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _print_frame(frame: Frame, print_pointcloud: bool, print_temperature: bool) -> None:
    if print_pointcloud:
        print(
            f"Frame {frame.header.frame_number}: "
            f"{frame.header.num_detected_obj} objects"
        )
        for p in frame.points:
            print(
                f"x: {p[0]:.3f} m, y: {p[1]:.3f} m, z: {p[2]:.3f} m, "
                f"v: {p[3]:.3f} m/s, snr: {int(p[4])}, noise: {int(p[5])}"
            )
    if print_temperature and frame.temperature is not None:
        t = frame.temperature
        print(
            f"Temperature [degC] rx: {t.rx0}/{t.rx1}/{t.rx2}/{t.rx3} "
            f"tx: {t.tx0}/{t.tx1}/{t.tx2} pm: {t.pm} dig: {t.dig0}/{t.dig1}"
        )


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
    if args.no_save:
        config.output.save_pointcloud = False
        config.output.save_temperature = False

    start_time = time()
    frame_count = 0

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            pointcloud_writer = None
            if config.output.save_pointcloud:
                pointcloud_writer = stack.enter_context(
                    PointCloudWriter(config.output.pointcloud_path)
                )
            temperature_writer = None
            if config.output.save_temperature:
                temperature_writer = stack.enter_context(
                    TemperatureWriter(config.output.temperature_path)
                )

            def handle_frame(frame: Frame) -> None:
                nonlocal frame_count
                frame_count += 1
                _print_frame(
                    frame,
                    config.display.print_pointcloud,
                    config.display.print_temperature,
                )
                if pointcloud_writer is not None:
                    pointcloud_writer.write(frame.points, frame.timestamp)
                if temperature_writer is not None:
                    temperature_writer.write(frame.temperature, frame.timestamp)

            if args.gui:
                if args.max_frames is not None or args.duration is not None:
                    log.warning(
                        "--max-frames/--duration are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.viewer import run_viewer

                run_viewer(config, session.frames(), on_frame=handle_frame)
            else:
                for frame in session.frames():
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

    log.info("Received %d frames in %.1f s", frame_count, time() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())

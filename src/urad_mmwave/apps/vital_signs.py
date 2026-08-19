"""Vital Signs with People Tracking application (uRAD Industrial).

Requires the TI Vital Signs with People Tracking firmware (distributed as a
release asset in the uRAD Industrial repository). The firmware tracks one
person and estimates their heart and breathing rates from the phase of the
radar return.

The UART output shares the packet framing and tracking TLVs of the 3D
People Tracking firmware (see :mod:`urad_mmwave.apps.people_tracking`) and
adds one TLV:

    1040 — vital signs (target id, range bin, breathing deviation,
           heart rate, breathing rate and both waveform buffers)

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.vital_signs import parse_frame

    with RadarSession(load_config("config_radar.json")) as session:
        for fields, payload, timestamp in session.packets():
            frame = parse_frame(payload, timestamp)
            if frame.vitals is not None:
                print(frame.vitals.heart_rate, frame.vitals.breathing_rate)
"""

from __future__ import annotations

import argparse
import logging
import statistics
import struct
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
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
    TLV_TARGET_INDEX,
    TLV_TARGET_LIST,
    PeopleTrackingFrame,
    _AppendWriter,
    _parse_point_cloud,
    _parse_targets,
)
from urad_mmwave.config import load_config
from urad_mmwave.radar import RadarSession

log = logging.getLogger(__name__)

TLV_VITAL_SIGNS = 1040

_WAVEFORM_SAMPLES = 15
# target id, range bin (uint16) + breathing deviation, heart rate,
# breathing rate and the two 15-sample waveform buffers (floats).
_VITALS_STRUCT = struct.Struct("<2H33f")

# Below this breathing deviation the firmware considers the person to be
# holding their breath; exactly zero means no measurement is available yet.
# Threshold taken from the TI Radar Toolbox visualizer.
HOLDING_BREATH_THRESHOLD = 0.02


@dataclass(frozen=True)
class VitalSigns:
    """Decoded vital signs TLV for the tracked person."""

    target_id: int
    range_bin: int
    breathing_deviation: float
    heart_rate: float  # beats per minute
    breathing_rate: float  # breaths per minute
    heart_waveform: tuple[float, ...]  # last 15 heart waveform samples
    breath_waveform: tuple[float, ...]  # last 15 breathing waveform samples


@dataclass
class VitalSignsFrame:
    """One decoded vital signs packet.

    Attributes:
        tracking: The people tracking part of the frame (point cloud,
            targets, target index, presence).
        vitals: Vital signs of the tracked person, if reported.
        timestamp: Host epoch time when the frame was received.
    """

    tracking: PeopleTrackingFrame
    vitals: VitalSigns | None = None
    timestamp: float = 0.0


def parse_vitals(body: bytes) -> VitalSigns | None:
    """Decode the body of one vital signs TLV (type 1040)."""
    if len(body) < _VITALS_STRUCT.size:
        log.warning("Vital signs TLV too short (%d bytes)", len(body))
        return None
    fields = _VITALS_STRUCT.unpack_from(body, 0)
    return VitalSigns(
        target_id=fields[0],
        range_bin=fields[1],
        breathing_deviation=fields[2],
        heart_rate=fields[3],
        breathing_rate=fields[4],
        heart_waveform=fields[5 : 5 + _WAVEFORM_SAMPLES],
        breath_waveform=fields[5 + _WAVEFORM_SAMPLES : 5 + 2 * _WAVEFORM_SAMPLES],
    )


def parse_frame(payload: bytes, timestamp: float = 0.0) -> VitalSignsFrame:
    """Decode the TLV payload of one vital signs packet."""
    tracking = PeopleTrackingFrame(timestamp=timestamp)
    frame = VitalSignsFrame(tracking=tracking, timestamp=timestamp)
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
        elif tlv_type == TLV_PRESENCE:
            if len(body) >= _PRESENCE_STRUCT.size:
                tracking.presence = _PRESENCE_STRUCT.unpack_from(body, 0)[0]
        elif tlv_type == TLV_VITAL_SIGNS:
            frame.vitals = parse_vitals(body)
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        cursor += tlv_length

    return frame


def patient_status(frame: VitalSignsFrame) -> str:
    """Classify the frame the way the TI visualizer does."""
    if not frame.tracking.targets:
        return "no patient"
    if frame.vitals is None or frame.vitals.breathing_deviation == 0.0:
        return "measuring"
    if frame.vitals.breathing_deviation < HOLDING_BREATH_THRESHOLD:
        return "holding breath"
    return "present"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urad-vital-signs",
        description="Heart and breathing rate measurement with a uRAD "
        "Industrial radar running the Vital Signs with People Tracking "
        "firmware.",
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
        help="Directory for VitalSigns/PointCloud/Targets files " "(default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live heart and breathing waveforms "
        "(requires: pip install urad-mmwave[gui])",
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
    # The TI visualizer smooths the heart rate with the median of the
    # latest 10 measurements; do the same for the printed value.
    recent_heart_rates: deque[float] = deque(maxlen=10)

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            writers = {}
            if not args.no_save:
                for name in ("VitalSigns", "PointCloud", "Targets"):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def handle_frame(frame: VitalSignsFrame) -> None:
                status = patient_status(frame)

                # The firmware keeps measuring at the last locked range bin
                # even when the tracker drops a person who sits perfectly
                # still, so feed the median whenever the measurement is
                # valid rather than only while a track is active.
                if (
                    frame.vitals is not None
                    and frame.vitals.breathing_deviation >= HOLDING_BREATH_THRESHOLD
                ):
                    recent_heart_rates.append(frame.vitals.heart_rate)

                if frame.vitals is not None and recent_heart_rates:
                    print(
                        f"patient: {status}  "
                        f"heart: {statistics.median(recent_heart_rates):.1f} bpm  "
                        f"breath: {frame.vitals.breathing_rate:.1f} rpm  "
                        f"(deviation: {frame.vitals.breathing_deviation:.3f})"
                    )
                else:
                    print(f"patient: {status}")

                if writers:
                    if frame.vitals is not None:
                        writers["VitalSigns"].write_row(
                            [
                                frame.vitals.target_id,
                                frame.vitals.range_bin,
                                frame.vitals.breathing_deviation,
                                frame.vitals.heart_rate,
                                frame.vitals.breathing_rate,
                            ],
                            frame.timestamp,
                        )
                    writers["PointCloud"].write_row(
                        [v for p in frame.tracking.points for v in p],
                        frame.timestamp,
                    )
                    writers["Targets"].write_row(
                        [
                            v
                            for t in frame.tracking.targets
                            for v in (t.tid, *t.position, *t.velocity)
                        ],
                        frame.timestamp,
                    )

            if args.gui:
                if args.duration is not None:
                    log.warning(
                        "--duration is ignored in GUI mode; close the window to stop"
                    )
                from urad_mmwave.apps.vital_signs_viewer import run_viewer

                run_viewer(
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

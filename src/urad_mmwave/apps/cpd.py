"""Child Presence Detection (CPD) with classification (uRAD Industrial).

Requires the TI CPD with Classification firmware
(``occupancy_detection_3d_68xx.bin``, distributed as a release asset in the
uRAD Industrial repository). The firmware streams a 3D point cloud over
UART; the zone mapping, the occupancy detection state machine and the
adult/child classification run on the host — this module reproduces the
processing of the TI MATLAB visualizer (Radar Toolbox 4.00.00.05).

The firmware uses its own 48-byte packet header (same sync word as the
out-of-box demo, but with timing fields and a 16-bit TLV count + checksum)
and a single TLV:

    6 — compressed spherical point cloud (units block + 8 bytes per point)

Unlike the people tracking firmwares, the TLV length field here INCLUDES
the 8-byte TLV header itself.

The cabin geometry (sensor mounting, seat zones as cuboids), the state
machine thresholds and the classification parameters are all read from the
chirp configuration file (``sensorPosition``, ``cuboidDef``,
``occStateMach``, ``zoneNeighDef``, ``classParam`` …).

Typical usage:

    from urad_mmwave import RadarSession, load_config
    from urad_mmwave.apps.cpd import (
        HEADER_FORMAT, OccupancyTracker, parse_frame, read_cabin_setup,
    )

    config = load_config("config_radar.json")
    config.packet.header_format = HEADER_FORMAT
    setup = read_cabin_setup(config.chirp_config_path)
    tracker = OccupancyTracker(setup)
    with RadarSession(config) as session:
        for fields, payload, timestamp in session.packets():
            frame = parse_frame(payload, timestamp, frame_number=fields[3])
            tracker.update(frame)
            print([zone.label for zone in frame.zones])
"""

from __future__ import annotations

import argparse
import logging
import math
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

TLV_POINT_CLOUD = 6

# CPD packet header: sync word + version, totalPacketLen, platform,
# frameNumber, subFrameNumber, chirpProcessingMargin,
# frameProcessingTimeInUsec, trackingProcessingTimeInUsec,
# uartSendingTimeInUsec (uint32 each) + numTLVs, checksum (uint16 each).
HEADER_FORMAT = "<Q9I2H"

MAX_TLV_TYPE = 20
MAX_TLV_LENGTH = 10000  # firmware maximum is 750 points -> 6028 bytes

# Reading 0xBE alignment padding as a TLV header yields this word (kept for
# robustness; the CPD firmware itself does not pad its packets).
PADDING_WORD = 0xBEBEBEBE

_TLV_HEADER = struct.Struct("<2I")
_POINT_UNIT = struct.Struct("<5f")  # elevation, azimuth, doppler, range, snr units
_POINT_STRUCT = struct.Struct("<2bh2H")  # elev int8, azim int8, dopp i16, rng/snr u16

# Classification decisions (as defined by the TI visualizer).
DECISION_NONE = 0
DECISION_CHILD = 1
DECISION_ADULT = 2

_OVERLOAD_FREEZE_FRAMES = 2  # frames all zones stay frozen after an overload


@dataclass(frozen=True)
class SensorPosition:
    """Sensor mounting offset and rotation (``sensorPosition`` command).

    Offsets are in meters in car coordinates (X to the driver side, Y to
    the rear, Z up from the floor); rotations are clockwise, in degrees.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yz_rot: float = 0.0
    xy_rot: float = 0.0
    xz_rot: float = 0.0


@dataclass(frozen=True)
class StateMachineParams:
    """Occupancy state machine thresholds for one zone type (``occStateMach``)."""

    enter_points_1: float  # points and avg SNR to enter occupied (condition 1)
    enter_snr_1: float
    enter_points_2: float  # points and avg SNR to enter occupied (condition 2)
    enter_snr_2: float
    entry_frames: int  # consecutive qualifying frames required to enter
    stay_points: float  # points and avg SNR to remain occupied
    stay_snr: float
    forget_frames: int  # miss frames tolerated before leaving occupied
    forget_points: float  # below this point count a frame counts as a miss
    overload_snr: float  # avg SNR that freezes the state machine (movement)


@dataclass(frozen=True)
class ClassificationParams:
    """Adult/child classification setup (``classParam`` command)."""

    enabled: bool
    num_frame_avg: int  # occupied frames accumulated per decision
    snr_thresholds: tuple[float, ...]  # per zone, dB (10*log10 of avg total SNR)
    volume_thresholds: tuple[float, ...]  # per zone, m^2 (sum of x/y/z variances)


@dataclass
class CabinZone:
    """One occupancy zone defined by up to three cuboids (``cuboidDef``)."""

    zone_id: int  # 1-based, as in the configuration file
    cuboids: list[tuple[float, float, float, float, float, float]] = field(
        default_factory=list
    )  # (xmin, xmax, ymin, ymax, zmin, zmax) in car coordinates, meters
    zone_type: int = 0  # selects the occStateMach parameter set
    neighbors: tuple[int, ...] = ()  # 1-based ids of neighbor zones

    @property
    def is_null(self) -> bool:
        """True for NULL zones (a single all-zero cuboid, never occupied)."""
        return all(value == 0.0 for cuboid in self.cuboids for value in cuboid)

    def bounds_xy(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) bounding rectangle over all cuboids."""
        xmin = min(c[0] for c in self.cuboids)
        xmax = max(c[1] for c in self.cuboids)
        ymin = min(c[2] for c in self.cuboids)
        ymax = max(c[3] for c in self.cuboids)
        return xmin, xmax, ymin, ymax


@dataclass
class CabinSetup:
    """Cabin geometry and host processing parameters from the chirp config."""

    num_zones: int
    zones: list[CabinZone]
    state_machines: dict[int, StateMachineParams]  # keyed by zone type
    sensor_position: SensorPosition
    classification: ClassificationParams
    tot_num_rows: int = 2
    interior_bounds: tuple[float, float, float, float] | None = None


@dataclass
class ZoneStatus:
    """Runtime occupancy state of one zone."""

    zone_id: int  # 1-based
    occupied: bool = False
    num_points: int = 0
    avg_snr: float = 0.0
    decision: int = DECISION_NONE  # DECISION_NONE / DECISION_CHILD / DECISION_ADULT
    decision_snr_db: float | None = None  # metrics behind the last decision
    decision_volume: float | None = None

    @property
    def label(self) -> str:
        """Human-readable state: empty / occupied / child / adult."""
        if not self.occupied:
            return "empty"
        if self.decision == DECISION_CHILD:
            return "child"
        if self.decision == DECISION_ADULT:
            return "adult"
        return "occupied"


@dataclass
class CpdFrame:
    """One decoded CPD frame.

    Attributes:
        points: Array of shape (N, 5) with columns range (m), azimuth (rad),
            elevation (rad), doppler (m/s), snr (linear, firmware units).
        cabin_points: Array of shape (N, 4) with columns x, y, z (m, car
            coordinates) and snr; filled by :meth:`OccupancyTracker.update`.
        zones: Per-zone occupancy/classification state; filled by
            :meth:`OccupancyTracker.update`.
        frame_number: Frame counter reported in the packet header.
        timestamp: Host epoch time when the frame was received.
    """

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    cabin_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    zones: list[ZoneStatus] = field(default_factory=list)
    frame_number: int = 0
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
            azimuth * azim_unit,
            elevation * elev_unit,
            doppler * doppler_unit,
            snr * snr_unit,
        )
    return points


def parse_frame(
    payload: bytes, timestamp: float = 0.0, frame_number: int = 0
) -> CpdFrame:
    """Decode the TLV payload of one CPD packet.

    The CPD firmware's TLV length field includes the 8-byte TLV header
    itself (unlike the people tracking firmwares), so the body is
    ``length - 8`` bytes.
    """
    frame = CpdFrame(timestamp=timestamp, frame_number=frame_number)
    cursor = 0

    while cursor + _TLV_HEADER.size <= len(payload):
        tlv_type, tlv_length = _TLV_HEADER.unpack_from(payload, cursor)
        cursor += _TLV_HEADER.size

        if tlv_type == PADDING_WORD:
            break  # end-of-frame alignment padding
        if (
            tlv_type > MAX_TLV_TYPE
            or tlv_length < _TLV_HEADER.size
            or tlv_length > MAX_TLV_LENGTH
        ):
            log.warning(
                "Implausible TLV (type=%d, length=%d); discarding rest of frame",
                tlv_type,
                tlv_length,
            )
            break

        body_length = tlv_length - _TLV_HEADER.size  # length includes the header
        if cursor + body_length > len(payload):
            log.warning(
                "TLV type %d declares %d bytes but only %d remain; discarding",
                tlv_type,
                body_length,
                len(payload) - cursor,
            )
            break

        body = payload[cursor : cursor + body_length]

        if tlv_type == TLV_POINT_CLOUD:
            frame.points = _parse_point_cloud(body)
        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, body_length)

        cursor += body_length

    return frame


def read_cabin_setup(chirp_config_path: str | Path) -> CabinSetup:
    """Extract the cabin geometry and host parameters from a chirp config.

    Parses the host-side commands of the CPD demo (``numZones``,
    ``totNumRows``, ``sensorPosition``, ``cuboidDef``, ``zoneNeighDef``,
    ``occStateMach``, ``interiorBounds``, ``classParam``) exactly as the TI
    visualizer does.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file defines no zones (not a CPD configuration).
    """
    from urad_mmwave.radar import read_chirp_config

    num_zones = 0
    tot_num_rows = 2
    sensor_position = SensorPosition()
    zones: dict[int, CabinZone] = {}
    state_machines: dict[int, StateMachineParams] = {}
    interior_bounds: tuple[float, float, float, float] | None = None
    class_mode = False
    class_frames = 1
    class_snr: tuple[float, ...] = ()
    class_vol: tuple[float, ...] = ()

    host_commands = (
        "numZones",
        "totNumRows",
        "sensorPosition",
        "cuboidDef",
        "zoneNeighDef",
        "occStateMach",
        "interiorBounds",
        "classParam",
    )
    for command in read_chirp_config(chirp_config_path):
        parts = command.split()
        name = parts[0]
        if name not in host_commands:
            continue
        try:
            args = [float(v) for v in parts[1:]]
        except ValueError:
            log.warning(
                "Skipping malformed command in %s: %s", chirp_config_path, command
            )
            continue

        if name == "numZones":
            num_zones = int(args[0])
        elif name == "totNumRows":
            tot_num_rows = int(args[0])
        elif name == "sensorPosition" and len(args) >= 6:
            sensor_position = SensorPosition(*args[:6])
        elif name == "cuboidDef" and len(args) >= 8:
            zone_id = int(args[0])
            zone = zones.setdefault(zone_id, CabinZone(zone_id=zone_id))
            zone.cuboids.append(tuple(args[2:8]))
        elif name == "zoneNeighDef" and len(args) >= 3:
            zone_id = int(args[0])
            zone = zones.setdefault(zone_id, CabinZone(zone_id=zone_id))
            zone.zone_type = int(args[1])
            num_neighbors = int(args[2])
            zone.neighbors = tuple(int(v) for v in args[3 : 3 + num_neighbors])
        elif name == "occStateMach" and len(args) >= 11:
            zone_type = int(args[0])
            state_machines[zone_type] = StateMachineParams(
                enter_points_1=args[1],
                enter_snr_1=args[2],
                enter_points_2=args[3],
                enter_snr_2=args[4],
                entry_frames=int(args[5]),
                stay_points=args[6],
                stay_snr=args[7],
                forget_frames=int(args[8]),
                forget_points=args[9],
                overload_snr=args[10],
            )
        elif name == "interiorBounds" and len(args) >= 4:
            interior_bounds = tuple(args[:4])
        elif name == "classParam" and len(args) >= 2:
            class_mode = bool(int(args[0]))
            class_frames = int(args[1])
            thresholds = args[2:]
            half = len(thresholds) // 2
            class_snr = tuple(thresholds[:half])
            class_vol = tuple(thresholds[half : 2 * half])

    if num_zones == 0 or not zones:
        raise ValueError(
            f"{chirp_config_path} defines no occupancy zones (numZones/"
            "cuboidDef commands missing) — is this a CPD chirp configuration?"
        )

    zone_list = [
        zones.get(zone_id, CabinZone(zone_id=zone_id))
        for zone_id in range(1, num_zones + 1)
    ]
    if not state_machines:
        raise ValueError(f"{chirp_config_path} has no occStateMach command")

    return CabinSetup(
        num_zones=num_zones,
        zones=zone_list,
        state_machines=state_machines,
        sensor_position=sensor_position,
        classification=ClassificationParams(
            enabled=class_mode,
            num_frame_avg=class_frames,
            snr_thresholds=class_snr,
            volume_thresholds=class_vol,
        ),
        tot_num_rows=tot_num_rows,
        interior_bounds=interior_bounds,
    )


def points_to_cabin(points: np.ndarray, sensor: SensorPosition) -> np.ndarray:
    """Transform spherical sensor points to cartesian car coordinates.

    ``points`` is the (N, 5) array of :class:`CpdFrame` (range m, azimuth
    rad, elevation rad, doppler, snr). Returns an (N, 4) array with columns
    x, y, z (meters, car coordinates: floor at z = 0) and snr, applying the
    mounting rotations and offsets exactly as the TI visualizer does.
    """
    if not len(points):
        return np.zeros((0, 4))
    range_m = points[:, 0]
    azimuth = points[:, 1]
    elevation = points[:, 2]
    snr = points[:, 4]

    x = range_m * np.cos(elevation) * np.sin(azimuth)
    y = range_m * np.cos(elevation) * np.cos(azimuth)
    z = range_m * np.sin(elevation)

    yz = math.radians(sensor.yz_rot)
    xy = math.radians(sensor.xy_rot)
    xz = math.radians(sensor.xz_rot)
    # Rotation in the y-z plane, then x-y, then x-z (clockwise, as MATLAB).
    y1 = math.cos(yz) * y + math.sin(yz) * z
    z1 = -math.sin(yz) * y + math.cos(yz) * z
    x1 = math.cos(xy) * x + math.sin(xy) * y1
    y2 = -math.sin(xy) * x + math.cos(xy) * y1
    x2 = math.cos(xz) * x1 + math.sin(xz) * z1
    z2 = -math.sin(xz) * x1 + math.cos(xz) * z1

    return np.column_stack(
        (x2 + sensor.x, y2 + sensor.y, z2 + sensor.z, snr)
    )


def assign_zones(cabin_points: np.ndarray, zones: list[CabinZone]) -> np.ndarray:
    """Map car-coordinate points to zones.

    Returns a boolean array of shape (N, num_zones); a point belongs to a
    zone when it falls strictly inside any of the zone's cuboids.
    """
    count = len(cabin_points)
    zone_map = np.zeros((count, len(zones)), dtype=bool)
    if count == 0:
        return zone_map
    x, y, z = cabin_points[:, 0], cabin_points[:, 1], cabin_points[:, 2]
    for column, zone in enumerate(zones):
        for xmin, xmax, ymin, ymax, zmin, zmax in zone.cuboids:
            inside = (
                (x > xmin) & (x < xmax)
                & (y > ymin) & (y < ymax)
                & (z > zmin) & (z < zmax)
            )
            zone_map[:, column] |= inside
    return zone_map


class _ZoneTracker:
    """Mutable per-zone state (occupancy counters and classification)."""

    def __init__(self) -> None:
        self.state = 0  # 0 = not occupied, 1 = occupied
        self.num_entry_count = 0
        self.detect_to_free_count = 0
        self.freeze = 0
        self.num_points = 0
        self.avg_snr = 0.0
        # Classification accumulators (occupied frames only).
        self.class_frames = 0
        self.class_x: list[float] = []
        self.class_y: list[float] = []
        self.class_z: list[float] = []
        self.class_snr_sum = 0.0
        self.decision = DECISION_NONE
        self.decision_snr_db: float | None = None
        self.decision_volume: float | None = None

    def reset_classification(self) -> None:
        self.class_frames = 0
        self.class_x = []
        self.class_y = []
        self.class_z = []
        self.class_snr_sum = 0.0


def _sample_variance(values: list[float]) -> float:
    """MATLAB-style sample variance (normalized by N-1; 0 for N < 2)."""
    if len(values) < 2:
        return 0.0
    return float(np.var(np.asarray(values), ddof=1))


class OccupancyTracker:
    """Host-side occupancy state machine and adult/child classifier.

    Faithful port of the TI visualizer processing (``occupancyDetection.m``
    and ``classification_logic_perSeat.m``): per-frame zone statistics,
    overload freeze, two-condition entry (with neighbor SNR comparison),
    hysteresis for leaving, and threshold-based classification over a
    sliding accumulation window of occupied frames.
    """

    def __init__(self, setup: CabinSetup):
        self._setup = setup
        self._trackers = [_ZoneTracker() for _ in setup.zones]

    def update(self, frame: CpdFrame) -> list[ZoneStatus]:
        """Process one frame: fills ``frame.cabin_points`` and ``frame.zones``."""
        setup = self._setup
        frame.cabin_points = points_to_cabin(frame.points, setup.sensor_position)
        zone_map = assign_zones(frame.cabin_points, setup.zones)
        snr = frame.cabin_points[:, 3] if len(frame.cabin_points) else np.zeros(0)

        for column, tracker in enumerate(self._trackers):
            members = zone_map[:, column]
            tracker.num_points = int(members.sum())
            tracker.avg_snr = float(snr[members].mean()) if tracker.num_points else 0.0

        # Overload check: excessive movement anywhere freezes every zone.
        # The TI visualizer compares against the zone type 0 threshold.
        overload_threshold = self._params_for_type(0).overload_snr
        for tracker in self._trackers:
            if tracker.avg_snr >= overload_threshold:
                for other in self._trackers:
                    other.freeze = _OVERLOAD_FREEZE_FRAMES
            elif tracker.freeze > 0:
                tracker.freeze -= 1

        for zone, tracker in zip(setup.zones, self._trackers):
            self._update_state_machine(zone, tracker)

        if setup.classification.enabled and len(frame.points):
            self._classify(frame.cabin_points, zone_map)

        frame.zones = [
            ZoneStatus(
                zone_id=zone.zone_id,
                occupied=tracker.state == 1,
                num_points=tracker.num_points,
                avg_snr=tracker.avg_snr,
                decision=tracker.decision,
                decision_snr_db=tracker.decision_snr_db,
                decision_volume=tracker.decision_volume,
            )
            for zone, tracker in zip(setup.zones, self._trackers)
        ]
        return frame.zones

    def _params_for_type(self, zone_type: int) -> StateMachineParams:
        try:
            return self._setup.state_machines[zone_type]
        except KeyError:
            return next(iter(self._setup.state_machines.values()))

    def _update_state_machine(self, zone: CabinZone, tracker: _ZoneTracker) -> None:
        if tracker.freeze != 0:
            return
        params = self._params_for_type(zone.zone_type)

        # Condition 2 additionally requires the zone to beat its neighbors'
        # average SNR (reduces false detections on the middle seat).
        max_avg_snr = params.enter_snr_2
        for neighbor_id in zone.neighbors:
            if 1 <= neighbor_id <= len(self._trackers):
                max_avg_snr = max(
                    max_avg_snr, self._trackers[neighbor_id - 1].avg_snr
                )

        if tracker.state == 0:  # NOT_OCCUPIED
            if (
                tracker.num_points > params.enter_points_1
                and tracker.avg_snr > params.enter_snr_1
            ) or (
                tracker.num_points > params.enter_points_2
                and tracker.avg_snr > max_avg_snr
            ):
                tracker.num_entry_count += 1
            else:
                tracker.num_entry_count = 0
            if tracker.num_entry_count >= params.entry_frames:
                tracker.state = 1
                tracker.detect_to_free_count = 0
        else:  # OCCUPIED
            if (
                tracker.num_points > params.stay_points
                and tracker.avg_snr > params.stay_snr
            ):
                tracker.detect_to_free_count = 0
            elif tracker.num_points < params.forget_points:
                if tracker.detect_to_free_count > params.forget_frames:
                    tracker.state = 0
                    tracker.num_entry_count = 0
                else:
                    tracker.detect_to_free_count += 1
            else:
                tracker.detect_to_free_count -= 1

    def _classify(self, cabin_points: np.ndarray, zone_map: np.ndarray) -> None:
        params = self._setup.classification
        for column, tracker in enumerate(self._trackers):
            if tracker.state == 1:
                members = zone_map[:, column]
                tracker.class_frames += 1
                tracker.class_x.extend(cabin_points[members, 0])
                tracker.class_y.extend(cabin_points[members, 1])
                tracker.class_z.extend(cabin_points[members, 2])
                tracker.class_snr_sum += float(cabin_points[members, 3].sum())

                if tracker.class_frames == params.num_frame_avg:
                    mean_snr = tracker.class_snr_sum / params.num_frame_avg
                    snr_db = 10 * math.log10(mean_snr) if mean_snr > 0 else -math.inf
                    volume = (
                        _sample_variance(tracker.class_x)
                        + _sample_variance(tracker.class_y)
                        + _sample_variance(tracker.class_z)
                    )
                    snr_threshold = _threshold(params.snr_thresholds, column)
                    vol_threshold = _threshold(params.volume_thresholds, column)
                    if snr_db < snr_threshold and volume < vol_threshold:
                        tracker.decision = DECISION_CHILD
                    else:
                        tracker.decision = DECISION_ADULT
                    tracker.decision_snr_db = snr_db
                    tracker.decision_volume = volume
                    tracker.reset_classification()
            elif tracker.class_frames != 0 or tracker.decision != DECISION_NONE:
                tracker.reset_classification()
                tracker.decision = DECISION_NONE
                tracker.decision_snr_db = None
                tracker.decision_volume = None


def _threshold(thresholds: tuple[float, ...], column: int) -> float:
    return thresholds[column] if column < len(thresholds) else math.inf


def format_zone_summary(zones: list[ZoneStatus]) -> str:
    """One-line console summary, e.g. ``Z1:adult Z2:empty Z3:child``."""
    return " ".join(f"Z{zone.zone_id}:{zone.label}" for zone in zones)


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
        prog="urad-cpd",
        description="In-cabin occupancy and child presence detection (CPD) "
        "with adult/child classification, using a uRAD Industrial radar "
        "running the CPD with Classification firmware.",
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
        help="Directory for PointCloud/Zones files (default: ./output)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable all file output"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live cabin view with the zone states and point cloud "
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

    # The CPD firmware uses its own 48-byte header, not the out-of-box one.
    config.packet.header_format = HEADER_FORMAT

    try:
        setup = read_cabin_setup(config.chirp_config_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    log.info(
        "Cabin setup: %d zones, %d row(s), classification %s",
        setup.num_zones,
        setup.tot_num_rows,
        "enabled" if setup.classification.enabled else "disabled",
    )

    tracker = OccupancyTracker(setup)
    output_dir = Path(args.output_dir)
    start_time = time()
    frame_count = 0

    try:
        with ExitStack() as stack:
            session = stack.enter_context(RadarSession(config))

            writers = {}
            if not args.no_save:
                for name in ("PointCloud", "Zones"):
                    writers[name] = stack.enter_context(
                        _AppendWriter(output_dir / f"{name}.txt")
                    )
                log.info("Writing output files to %s", output_dir)

            def frames():
                for fields, payload, timestamp in session.packets():
                    frame = parse_frame(payload, timestamp, frame_number=fields[3])
                    tracker.update(frame)
                    yield frame

            def handle_frame(frame: CpdFrame) -> None:
                occupied = sum(zone.occupied for zone in frame.zones)
                print(
                    f"{format_zone_summary(frame.zones)}  "
                    f"points: {len(frame.points)}  occupied: {occupied}"
                )

                if writers:
                    writers["PointCloud"].write_row(
                        [v for p in frame.cabin_points for v in p], frame.timestamp
                    )
                    writers["Zones"].write_row(
                        [
                            v
                            for zone in frame.zones
                            for v in (
                                zone.zone_id,
                                int(zone.occupied),
                                zone.num_points,
                                zone.avg_snr,
                                zone.decision,
                            )
                        ],
                        frame.timestamp,
                    )

            if args.gui:
                if args.duration is not None or args.max_frames is not None:
                    log.warning(
                        "--duration/--max-frames are ignored in GUI mode; "
                        "close the window to stop"
                    )
                from urad_mmwave.apps.cpd_viewer import run_viewer

                run_viewer(setup, frames(), on_frame=handle_frame)
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

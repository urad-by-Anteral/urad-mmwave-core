"""TLV stream parser for the TI mmWave out-of-box demo (SDK 3.x).

Reads the binary UART stream produced by the radar, synchronizes on the
magic word, and decodes each frame into a :class:`Frame` with a point cloud
array and optional temperature report.

Supported TLV types:
    1  — detected points (x, y, z, v as float32)
    7  — side info per point (snr, noise as uint16)
    9  — temperature statistics
Any other TLV type is skipped by its declared length, keeping the parser
aligned with the stream.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from time import time
from typing import Protocol

import numpy as np

from urad_mmwave.config import PacketConfig

log = logging.getLogger(__name__)

TLV_DETECTED_POINTS = 1
TLV_STATS = 6
TLV_SIDE_INFO = 7
TLV_TEMPERATURE = 9

_POINT_STRUCT = struct.Struct("<4f")
_SIDE_INFO_STRUCT = struct.Struct("<2H")
_TEMPERATURE_STRUCT = struct.Struct("<iI10h")

# Upper bound for a plausible total packet length; anything above this after a
# valid sync word is treated as corruption and re-synchronized.
_MAX_PACKET_LEN = 65536


class SerialLike(Protocol):
    """Minimal read interface shared by pyserial ports and test doubles."""

    def read(self, size: int) -> bytes: ...


class StreamTimeoutError(TimeoutError):
    """Raised when the stream stops delivering data mid-packet."""


@dataclass(frozen=True)
class FrameHeader:
    """Decoded mmWave frame header (fields after the sync word)."""

    version: int
    total_packet_len: int
    platform: int
    frame_number: int
    time_cpu_cycles: int
    num_detected_obj: int
    num_tlvs: int
    subframe_number: int


@dataclass(frozen=True)
class TemperatureReport:
    """On-chip temperature sensors report (TLV type 9), values in °C."""

    valid: int
    time_ms: int
    rx0: int
    rx1: int
    rx2: int
    rx3: int
    tx0: int
    tx1: int
    tx2: int
    pm: int
    dig0: int
    dig1: int


@dataclass
class Frame:
    """One decoded radar frame.

    Attributes:
        header: Decoded frame header.
        points: Array of shape (num_detected_obj, 6) with columns
            x, y, z, v, snr, noise.
        temperature: Temperature report if the frame carried TLV type 9.
        timestamp: Host epoch time when the frame header was received.
    """

    header: FrameHeader
    points: np.ndarray
    temperature: TemperatureReport | None = None
    timestamp: float = 0.0


def read_exact(port: SerialLike, size: int, max_empty_reads: int = 50) -> bytes:
    """Read exactly ``size`` bytes from ``port``.

    Raises:
        StreamTimeoutError: If ``max_empty_reads`` consecutive reads return
            no data (the port timeout bounds the duration of each read).
    """
    buf = bytearray()
    empty_reads = 0
    while len(buf) < size:
        chunk = port.read(size - len(buf))
        if chunk:
            empty_reads = 0
            buf += chunk
        else:
            empty_reads += 1
            if empty_reads >= max_empty_reads:
                raise StreamTimeoutError(
                    f"Expected {size} bytes but got {len(buf)} before timing out"
                )
    return bytes(buf)


def parse_tlvs(
    payload: bytes, header: FrameHeader, config: PacketConfig
) -> tuple[np.ndarray, TemperatureReport | None]:
    """Parse the TLV payload of one frame.

    Returns the (num_detected_obj, 6) point array and the temperature report
    if present. Unknown TLV types are skipped by their declared length; a
    malformed or truncated TLV aborts the rest of the frame with a warning
    instead of desynchronizing the stream.
    """
    tlv_header = struct.Struct(config.tlv_header_format)
    points = np.zeros((header.num_detected_obj, 6))
    temperature: TemperatureReport | None = None
    offset = 0

    for _ in range(header.num_tlvs):
        if offset + tlv_header.size > len(payload):
            log.warning("Frame payload truncated before next TLV header")
            break

        tlv_type, tlv_length = tlv_header.unpack_from(payload, offset)
        offset += tlv_header.size

        if tlv_type > config.max_tlv_type or tlv_length > config.max_tlv_length:
            log.warning(
                "Implausible TLV (type=%d, length=%d); discarding rest of frame",
                tlv_type,
                tlv_length,
            )
            break

        if offset + tlv_length > len(payload):
            log.warning(
                "TLV type %d declares %d bytes but only %d remain; discarding",
                tlv_type,
                tlv_length,
                len(payload) - offset,
            )
            break

        body = payload[offset : offset + tlv_length]

        if tlv_type == TLV_DETECTED_POINTS:
            count = min(header.num_detected_obj, len(body) // _POINT_STRUCT.size)
            for j in range(count):
                points[j, :4] = _POINT_STRUCT.unpack_from(body, j * _POINT_STRUCT.size)

        elif tlv_type == TLV_SIDE_INFO:
            count = min(header.num_detected_obj, len(body) // _SIDE_INFO_STRUCT.size)
            for j in range(count):
                points[j, 4:6] = _SIDE_INFO_STRUCT.unpack_from(
                    body, j * _SIDE_INFO_STRUCT.size
                )

        elif tlv_type == TLV_TEMPERATURE:
            if len(body) >= _TEMPERATURE_STRUCT.size:
                temperature = TemperatureReport(
                    *_TEMPERATURE_STRUCT.unpack_from(body, 0)
                )
            else:
                log.warning("Temperature TLV too short (%d bytes)", len(body))

        else:
            log.debug("Skipping TLV type %d (%d bytes)", tlv_type, tlv_length)

        offset += tlv_length

    return points, temperature


def read_frames(port: SerialLike, config: PacketConfig) -> Iterator[Frame]:
    """Continuously read, synchronize and decode frames from ``port``.

    Yields one :class:`Frame` per valid packet. Garbage between packets is
    discarded byte by byte until the sync pattern is found again; packets
    that time out mid-read are dropped with a warning.
    """
    header_struct = struct.Struct(config.header_format)
    header_len = header_struct.size
    buf = bytearray()

    while True:
        chunk = port.read(header_len - len(buf))
        buf += chunk
        if len(buf) < header_len:
            continue

        fields = header_struct.unpack(bytes(buf[:header_len]))
        if fields[0] != config.sync_pattern:
            del buf[0]  # shift one byte and retry: re-synchronization
            continue

        header = FrameHeader(*fields[1:])
        timestamp = time()
        buf.clear()

        payload_len = header.total_packet_len - header_len
        if payload_len < 0 or header.total_packet_len > _MAX_PACKET_LEN:
            log.warning(
                "Implausible packet length %d; re-synchronizing",
                header.total_packet_len,
            )
            continue

        try:
            payload = read_exact(port, payload_len)
        except StreamTimeoutError as exc:
            log.warning("Dropped frame %d: %s", header.frame_number, exc)
            continue

        points, temperature = parse_tlvs(payload, header, config)
        yield Frame(
            header=header,
            points=points,
            temperature=temperature,
            timestamp=timestamp,
        )

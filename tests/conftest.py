"""Shared test helpers: synthetic mmWave packets and serial port doubles."""

from __future__ import annotations

import struct

import pytest

from urad_mmwave.config import DEFAULT_SYNC_PATTERN, PacketConfig

HEADER_STRUCT = struct.Struct("<Q8I")


class FakePort:
    """Serial port double that serves a fixed byte stream.

    Reading past the end raises AssertionError so a buggy parser fails the
    test instead of hanging in an infinite loop.
    """

    def __init__(self, data: bytes, chunk_size: int = 0):
        self._data = bytearray(data)
        self._chunk_size = chunk_size  # 0 = serve whatever is requested

    def read(self, size: int) -> bytes:
        assert self._data, "FakePort exhausted: parser read past end of stream"
        if self._chunk_size:
            size = min(size, self._chunk_size)
        chunk = bytes(self._data[:size])
        del self._data[:size]
        return chunk


class EmptyPort:
    """Serial port double that always times out (returns no data)."""

    def read(self, size: int) -> bytes:
        return b""


def build_tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def build_points_tlv(points: list[tuple[float, float, float, float]]) -> bytes:
    body = b"".join(struct.pack("<4f", *p) for p in points)
    return build_tlv(1, body)


def build_side_info_tlv(side_info: list[tuple[int, int]]) -> bytes:
    body = b"".join(struct.pack("<2H", *s) for s in side_info)
    return build_tlv(7, body)


def build_temperature_tlv(values: tuple[int, ...]) -> bytes:
    return build_tlv(9, struct.pack("<iI10h", *values))


def build_packet(
    tlvs: list[bytes],
    num_detected_obj: int,
    sync: int = DEFAULT_SYNC_PATTERN,
    frame_number: int = 1,
) -> bytes:
    payload = b"".join(tlvs)
    total_len = HEADER_STRUCT.size + len(payload)
    header = HEADER_STRUCT.pack(
        sync,  # magic word
        0x3060000,  # version
        total_len,
        0xA6843,  # platform
        frame_number,
        123456,  # time cpu cycles
        num_detected_obj,
        len(tlvs),
        0,  # subframe number
    )
    return header + payload


@pytest.fixture
def packet_config() -> PacketConfig:
    return PacketConfig()

"""Unit tests for the level sensing application (command builder + decoder)."""

from __future__ import annotations

import struct

import pytest

from urad_mmwave.apps.level_sensing import (
    HEADER_FORMAT,
    LevelSensingSettings,
    build_commands,
    chirp_slope,
    decode_ranges,
)


def _ranges_payload(r1: float, r2: float, r3: float) -> bytes:
    """Build a synthetic ranges TLV using the firmware's fixed-point format."""

    def encode(value: float) -> tuple[int, int]:
        fixed = int(value * 2**20)
        return fixed >> 16, fixed & 0xFFFF

    hi1, lo1 = encode(r1)
    hi2, lo2 = encode(r2)
    hi3, lo3 = encode(r3)
    descriptor = struct.pack("<2H", 3, 9)
    body = descriptor + struct.pack("<3H3h", lo1, lo3, lo2, hi1, hi2, hi3)
    return struct.pack("<2I", 1, len(body)) + body


def test_header_format_is_36_bytes():
    assert struct.calcsize(HEADER_FORMAT) == 36


def test_chirp_slope_clamping():
    assert chirp_slope(5) == 31.23  # below 12 m -> maximum slope
    assert chirp_slope(200) == 1.87  # above 150 m -> minimum slope
    assert chirp_slope(100) == pytest.approx(3.747405725)


def test_build_commands_start_frequency_by_model():
    awr = build_commands(LevelSensingSettings(model="AWR"))
    iwr = build_commands(LevelSensingSettings(model="IWR"))
    assert any(cmd.startswith("profileCfg 0 77 ") for cmd in awr)
    assert any(cmd.startswith("profileCfg 0 60 ") for cmd in iwr)
    assert awr[0] == "flushCfg"
    assert awr[-1] == "sensorStart"


def test_build_commands_range_limit_swapped_and_clamped():
    commands = build_commands(
        LevelSensingSettings(model="IWR", range_min=8.0, range_max=-2.0)
    )
    # min > max -> swapped; negative min -> clamped to 0.
    assert "RangeLimitCfg 2 1 0.0 8.0" in commands


def test_invalid_model_rejected():
    with pytest.raises(ValueError, match="model"):
        LevelSensingSettings(model="XWR")


def test_decode_ranges_roundtrip():
    payload = _ranges_payload(5.0, 7.25, 0.5)
    ranges = decode_ranges(payload)
    assert ranges == pytest.approx((5.0, 7.25, 0.5), abs=1e-4)


def test_decode_ranges_with_low_word_msb_set():
    # A value whose low 16 fixed-point bits are >= 0x8000 — the legacy
    # signed decode of r3_low read these 62.5 mm short.
    value = (3 * 2**16 + 0xC000) / 2**20  # low word = 0xC000
    payload = _ranges_payload(value, value, value)
    ranges = decode_ranges(payload)
    assert ranges == pytest.approx((value, value, value), abs=1e-6)


def test_decode_ranges_with_offset():
    payload = _ranges_payload(5.0, 5.0, 5.0)
    ranges = decode_ranges(payload, offset=0.25)
    assert ranges == pytest.approx((5.25, 5.25, 5.25), abs=1e-4)


def test_decode_ranges_skips_other_tlvs():
    stats = struct.pack("<2I", 6, 24) + b"\x00" * 24
    payload = stats + _ranges_payload(3.0, 3.0, 3.0)
    ranges = decode_ranges(payload)
    assert ranges == pytest.approx((3.0, 3.0, 3.0), abs=1e-4)


def test_decode_ranges_returns_none_without_tlv1():
    stats = struct.pack("<2I", 6, 24) + b"\x00" * 24
    assert decode_ranges(stats) is None

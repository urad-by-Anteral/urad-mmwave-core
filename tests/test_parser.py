"""Unit tests for the TLV stream parser, built on synthetic packets."""

from __future__ import annotations

from itertools import islice

import numpy as np
import pytest

from tests.conftest import (
    EmptyPort,
    FakePort,
    build_packet,
    build_points_tlv,
    build_side_info_tlv,
    build_temperature_tlv,
    build_tlv,
)
from urad_mmwave.parser import StreamTimeoutError, read_exact, read_frames

POINTS = [(1.0, 2.0, 3.0, 0.5), (-1.5, 4.0, 0.25, -0.75)]
SIDE_INFO = [(120, 30), (95, 42)]


def _read_one_frame(data: bytes, config, chunk_size: int = 0):
    port = FakePort(data, chunk_size=chunk_size)
    return next(iter(read_frames(port, config)))


def test_parses_points_and_side_info_paired(packet_config):
    packet = build_packet(
        [build_points_tlv(POINTS), build_side_info_tlv(SIDE_INFO)],
        num_detected_obj=2,
    )
    frame = _read_one_frame(packet, packet_config)

    assert frame.header.num_detected_obj == 2
    assert frame.header.num_tlvs == 2
    np.testing.assert_allclose(frame.points[0], [1.0, 2.0, 3.0, 0.5, 120, 30])
    np.testing.assert_allclose(frame.points[1], [-1.5, 4.0, 0.25, -0.75, 95, 42])
    assert frame.temperature is None
    assert frame.timestamp > 0


def test_side_info_before_points_still_paired(packet_config):
    packet = build_packet(
        [build_side_info_tlv(SIDE_INFO), build_points_tlv(POINTS)],
        num_detected_obj=2,
    )
    frame = _read_one_frame(packet, packet_config)
    np.testing.assert_allclose(frame.points[0], [1.0, 2.0, 3.0, 0.5, 120, 30])


def test_unknown_tlv_is_skipped_without_desync(packet_config):
    # A range-profile TLV (type 2) the parser does not decode must be skipped
    # by its declared length so the following TLVs parse correctly.
    unknown = build_tlv(2, b"\xab" * 64)
    packet = build_packet(
        [unknown, build_points_tlv(POINTS), build_side_info_tlv(SIDE_INFO)],
        num_detected_obj=2,
    )
    frame = _read_one_frame(packet, packet_config)
    np.testing.assert_allclose(frame.points[1], [-1.5, 4.0, 0.25, -0.75, 95, 42])


def test_resynchronizes_after_garbage(packet_config):
    packet = build_packet([build_points_tlv(POINTS)], num_detected_obj=2)
    data = b"\xde\xad\xbe\xef\x00" + packet
    frame = _read_one_frame(data, packet_config)
    assert frame.header.frame_number == 1
    np.testing.assert_allclose(frame.points[0, :4], [1.0, 2.0, 3.0, 0.5])


def test_multiple_frames_in_stream(packet_config):
    stream = build_packet(
        [build_points_tlv(POINTS)], num_detected_obj=2, frame_number=1
    ) + build_packet([build_points_tlv(POINTS)], num_detected_obj=2, frame_number=2)
    frames = list(islice(read_frames(FakePort(stream), packet_config), 2))
    assert [f.header.frame_number for f in frames] == [1, 2]


def test_chunked_reads(packet_config):
    # The serial port may return fewer bytes than requested on every read.
    packet = build_packet(
        [build_points_tlv(POINTS), build_side_info_tlv(SIDE_INFO)],
        num_detected_obj=2,
    )
    frame = _read_one_frame(packet, packet_config, chunk_size=7)
    np.testing.assert_allclose(frame.points[0], [1.0, 2.0, 3.0, 0.5, 120, 30])


def test_temperature_tlv(packet_config):
    values = (1, 5000, 40, 41, 42, 43, 50, 51, 52, 45, 60, 61)
    packet = build_packet(
        [build_points_tlv([]), build_temperature_tlv(values)],
        num_detected_obj=0,
    )
    frame = _read_one_frame(packet, packet_config)
    assert frame.temperature is not None
    assert frame.temperature.valid == 1
    assert frame.temperature.rx0 == 40
    assert frame.temperature.dig1 == 61
    assert frame.points.shape == (0, 6)


def test_truncated_tlv_does_not_crash(packet_config):
    # TLV declares more bytes than the payload actually carries.
    bad_tlv = build_tlv(1, b"\x00" * 8)[:12]  # header says 8 bytes, only 4 present
    packet = build_packet([bad_tlv], num_detected_obj=1)
    # Rebuild with a consistent total length for the truncated payload.
    frame = _read_one_frame(packet, packet_config)
    assert frame.points.shape == (1, 6)
    np.testing.assert_allclose(frame.points[0], np.zeros(6))


def test_implausible_tlv_aborts_frame(packet_config):
    bad = build_tlv(99, b"\x00" * 4)  # type above max_tlv_type
    packet = build_packet([bad, build_points_tlv(POINTS)], num_detected_obj=2)
    frame = _read_one_frame(packet, packet_config)
    # Rest of the frame is discarded, but the stream does not crash.
    np.testing.assert_allclose(frame.points, np.zeros((2, 6)))


def test_read_exact_times_out():
    with pytest.raises(StreamTimeoutError):
        read_exact(EmptyPort(), 10, max_empty_reads=3)

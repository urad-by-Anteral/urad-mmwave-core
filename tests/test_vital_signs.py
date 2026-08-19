"""Unit tests for the vital signs TLV parser, built on synthetic packets."""

from __future__ import annotations

import struct

import pytest

from urad_mmwave.apps.people_tracking import TLV_TARGET_LIST
from urad_mmwave.apps.vital_signs import (
    HOLDING_BREATH_THRESHOLD,
    TLV_VITAL_SIGNS,
    parse_frame,
    patient_status,
)


def _tlv(tlv_type: int, body: bytes) -> bytes:
    return struct.pack("<2I", tlv_type, len(body)) + body


def _vitals_tlv(
    target_id: int = 1,
    range_bin: int = 12,
    breathing_deviation: float = 0.1,
    heart_rate: float = 72.5,
    breathing_rate: float = 15.0,
) -> bytes:
    heart_waveform = [0.1 * i for i in range(15)]
    breath_waveform = [0.2 * i for i in range(15)]
    body = struct.pack(
        "<2H33f",
        target_id,
        range_bin,
        breathing_deviation,
        heart_rate,
        breathing_rate,
        *heart_waveform,
        *breath_waveform,
    )
    return _tlv(TLV_VITAL_SIGNS, body)


def _target_tlv(tid: int = 1) -> bytes:
    body = struct.pack("<I9f16f2f", tid, *([0.0] * 9), *([0.0] * 16), 1.0, 95.0)
    return _tlv(TLV_TARGET_LIST, body)


def test_parse_vitals_fields():
    frame = parse_frame(_target_tlv() + _vitals_tlv(), timestamp=2.0)

    assert frame.timestamp == 2.0
    assert len(frame.tracking.targets) == 1
    vitals = frame.vitals
    assert vitals is not None
    assert vitals.target_id == 1
    assert vitals.range_bin == 12
    assert vitals.breathing_deviation == pytest.approx(0.1)
    assert vitals.heart_rate == pytest.approx(72.5)
    assert vitals.breathing_rate == pytest.approx(15.0)
    assert len(vitals.heart_waveform) == 15
    assert len(vitals.breath_waveform) == 15
    assert vitals.heart_waveform[3] == pytest.approx(0.3)
    assert vitals.breath_waveform[10] == pytest.approx(2.0)


def test_frame_without_vitals_tlv():
    frame = parse_frame(_target_tlv())

    assert frame.vitals is None
    assert patient_status(frame) == "measuring"


def test_truncated_vitals_tlv_is_rejected():
    body = struct.pack("<2H3f", 1, 12, 0.1, 72.5, 15.0)  # waveforms missing
    frame = parse_frame(_tlv(TLV_VITAL_SIGNS, body))

    assert frame.vitals is None


def test_patient_status_classification():
    no_target = parse_frame(_vitals_tlv())
    assert patient_status(no_target) == "no patient"

    measuring = parse_frame(_target_tlv() + _vitals_tlv(breathing_deviation=0.0))
    assert patient_status(measuring) == "measuring"

    holding = parse_frame(
        _target_tlv() + _vitals_tlv(breathing_deviation=HOLDING_BREATH_THRESHOLD / 2)
    )
    assert patient_status(holding) == "holding breath"

    present = parse_frame(_target_tlv() + _vitals_tlv(breathing_deviation=0.1))
    assert patient_status(present) == "present"

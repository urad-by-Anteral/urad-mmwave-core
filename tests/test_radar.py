"""Unit tests for RadarSession configuration, using serial port doubles."""

from __future__ import annotations

import pytest

import urad_mmwave.radar as radar_module
from urad_mmwave.config import AppConfig, SerialConfig


class _ScriptedControlPort:
    """Control port double that replies to each written command."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._pending = b""

    def write(self, data: bytes) -> None:
        self._pending = self._responses.pop(0).encode() if self._responses else b""

    @property
    def in_waiting(self) -> int:
        return len(self._pending)

    def read(self, size: int) -> bytes:
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _config(tmp_path, commands: list[str]) -> AppConfig:
    chirp = tmp_path / "chirp.cfg"
    chirp.write_text("\n".join(commands) + "\n")
    return AppConfig(
        control_serial=SerialConfig(port="COM_TEST", baudrate=115200, timeout=0.01),
        data_serial=SerialConfig(port="COM_TEST", baudrate=921600, timeout=0.01),
        chirp_config_path=str(chirp),
    )


def test_configure_raises_when_radar_never_responds(tmp_path, monkeypatch):
    port = _ScriptedControlPort(responses=[])
    monkeypatch.setattr(radar_module, "_open_port", lambda config: port)
    session = radar_module.RadarSession(_config(tmp_path, ["sensorStop", "flushCfg"]))

    with pytest.raises(RuntimeError, match="power-cycle"):
        session._configure(["sensorStop", "flushCfg"])


def test_configure_accepts_responding_radar(tmp_path, monkeypatch):
    port = _ScriptedControlPort(responses=["Done", "Done"])
    monkeypatch.setattr(radar_module, "_open_port", lambda config: port)
    session = radar_module.RadarSession(_config(tmp_path, ["sensorStop", "flushCfg"]))

    session._configure(["sensorStop", "flushCfg"])  # must not raise

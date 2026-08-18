"""Dataclass-based configuration for the uRAD mmWave SDK.

Loads and validates the JSON configuration file that defines serial ports,
packet format, chirp configuration path and output options. Every value in
the JSON file is honoured at runtime (ports, baudrates, timeouts, paths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SYNC_PATTERN = 0x708050603040102
DEFAULT_HEADER_FORMAT = "<Q8I"
DEFAULT_TLV_HEADER_FORMAT = "<2I"


@dataclass
class SerialConfig:
    """Serial port parameters for one UART link."""

    port: str
    baudrate: int
    timeout: float = 0.3


@dataclass
class PacketConfig:
    """Binary packet structure and sanity limits (mmWave SDK 3.x defaults)."""

    sync_pattern: int = DEFAULT_SYNC_PATTERN
    header_format: str = DEFAULT_HEADER_FORMAT
    tlv_header_format: str = DEFAULT_TLV_HEADER_FORMAT
    max_tlv_type: int = 20
    max_tlv_length: int = 10000


@dataclass
class OutputConfig:
    """File output options."""

    save_pointcloud: bool = True
    pointcloud_path: str = "./output/PointCloud.txt"
    save_temperature: bool = False
    temperature_path: str = "./output/Temperature.txt"


@dataclass
class DisplayConfig:
    """Console printing options."""

    print_pointcloud: bool = True
    print_temperature: bool = False


@dataclass
class AppConfig:
    """Root configuration model for a uRAD mmWave radar session."""

    control_serial: SerialConfig
    data_serial: SerialConfig
    chirp_config_path: str = "./config/chirp_config.cfg"
    gpio_reset_pin: int | None = None
    packet: PacketConfig = field(default_factory=PacketConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @property
    def is_single_port(self) -> bool:
        """True when control and data share one physical UART (e.g. Raspberry Pi)."""
        return self.control_serial.port == self.data_serial.port


def _build_section(cls: type, data: dict[str, Any], section: str) -> Any:
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(
            f"Invalid '{section}' section in configuration: {exc}"
        ) from exc


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a JSON configuration file into an :class:`AppConfig`.

    The ``packet``, ``output`` and ``display`` sections are optional and fall
    back to mmWave SDK 3.x defaults. ``sync_pattern`` may be given as a hex
    string (``"0x708050603040102"``) or as an integer.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file contains unknown or missing keys.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fp:
        data = json.load(fp)

    for key in ("control_serial", "data_serial"):
        if key not in data:
            raise ValueError(f"Missing required '{key}' section in {config_path}")

    packet_data = data.get("packet", {})
    sync = packet_data.get("sync_pattern")
    if isinstance(sync, str):
        packet_data["sync_pattern"] = int(sync, 16)

    return AppConfig(
        control_serial=_build_section(
            SerialConfig, data["control_serial"], "control_serial"
        ),
        data_serial=_build_section(SerialConfig, data["data_serial"], "data_serial"),
        chirp_config_path=data.get("chirp_config_path", "./config/chirp_config.cfg"),
        gpio_reset_pin=data.get("gpio_reset_pin"),
        packet=_build_section(PacketConfig, packet_data, "packet"),
        output=_build_section(OutputConfig, data.get("output", {}), "output"),
        display=_build_section(DisplayConfig, data.get("display", {}), "display"),
    )

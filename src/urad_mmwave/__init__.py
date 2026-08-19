"""Core Python SDK for uRAD mmWave radar sensors by Anteral.

Provides serial communication, chirp configuration and TLV stream parsing
for uRAD products based on Texas Instruments AWR/IWR mmWave chips running
the out-of-box demo firmware (mmWave SDK 3.x).

Typical usage:

    from urad_mmwave import RadarSession, load_config

    config = load_config("config_radar.json")
    with RadarSession(config) as session:
        for frame in session.frames():
            print(frame.points)
"""

from __future__ import annotations

from urad_mmwave.config import (
    AppConfig,
    DisplayConfig,
    GuiConfig,
    OutputConfig,
    PacketConfig,
    SerialConfig,
    load_config,
)
from urad_mmwave.parser import (
    Frame,
    FrameHeader,
    StreamTimeoutError,
    TemperatureReport,
    iter_packets,
    read_frames,
)
from urad_mmwave.radar import RadarSession, read_chirp_config
from urad_mmwave.writer import PointCloudWriter, TemperatureWriter

__version__ = "0.2.1"

__all__ = [
    "AppConfig",
    "DisplayConfig",
    "Frame",
    "FrameHeader",
    "GuiConfig",
    "OutputConfig",
    "PacketConfig",
    "PointCloudWriter",
    "RadarSession",
    "SerialConfig",
    "StreamTimeoutError",
    "TemperatureReport",
    "TemperatureWriter",
    "iter_packets",
    "load_config",
    "read_chirp_config",
    "read_frames",
    "__version__",
]

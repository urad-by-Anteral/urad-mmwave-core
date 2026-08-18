"""Integration tests: every shipped product profile must load and be usable."""

from __future__ import annotations

from pathlib import Path

import pytest

from urad_mmwave.config import load_config
from urad_mmwave.radar import read_chirp_config

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
PROFILES = ["automotive", "automotive-hpa", "industrial"]


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_config_loads(profile):
    config = load_config(PROFILES_DIR / profile / "config_radar.json")
    assert config.control_serial.baudrate == 115200
    assert config.data_serial.baudrate == 921600
    assert len(config.gui.x_range) == 2
    assert len(config.gui.y_range) == 2


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_chirp_config_exists_and_parses(profile):
    config = load_config(PROFILES_DIR / profile / "config_radar.json")
    # The chirp path is resolved relative to the profile directory.
    assert Path(config.chirp_config_path).exists()

    commands = read_chirp_config(config.chirp_config_path)
    assert commands[0] == "sensorStop"
    assert commands[-1] == "sensorStart"
    assert not any(cmd.startswith("%") for cmd in commands)


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_temperature_chirp_exists(profile):
    temp_cfg = PROFILES_DIR / profile / "chirp_config_temperature.cfg"
    assert temp_cfg.exists()
    commands = read_chirp_config(temp_cfg)
    assert commands[0] == "sensorStop"

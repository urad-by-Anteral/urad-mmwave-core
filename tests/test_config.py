"""Unit tests for JSON configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urad_mmwave.config import DEFAULT_SYNC_PATTERN, load_config

FULL_CONFIG = {
    "control_serial": {"port": "COM8", "baudrate": 115200, "timeout": 0.3},
    "data_serial": {"port": "COM7", "baudrate": 921600, "timeout": 0.3},
    "chirp_config_path": "./config/chirp_config.cfg",
    "packet": {
        "sync_pattern": "0x708050603040102",
        "header_format": "<Q8I",
        "tlv_header_format": "<2I",
        "max_tlv_type": 20,
        "max_tlv_length": 10000,
    },
    "output": {
        "save_pointcloud": True,
        "pointcloud_path": "./output/PointCloud.txt",
    },
    "display": {"print_pointcloud": False},
}


def write_config(tmp_path, data):
    path = tmp_path / "config_radar.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_full_config(tmp_path):
    config = load_config(write_config(tmp_path, FULL_CONFIG))
    assert config.control_serial.port == "COM8"
    assert config.control_serial.baudrate == 115200
    assert config.data_serial.baudrate == 921600
    assert config.packet.sync_pattern == DEFAULT_SYNC_PATTERN
    assert config.output.save_pointcloud is True
    assert config.display.print_pointcloud is False
    assert config.is_single_port is False


def test_minimal_config_uses_defaults(tmp_path):
    minimal = {
        "control_serial": {"port": "COM8", "baudrate": 115200},
        "data_serial": {"port": "COM7", "baudrate": 921600},
    }
    config = load_config(write_config(tmp_path, minimal))
    assert config.packet.sync_pattern == DEFAULT_SYNC_PATTERN
    assert config.packet.max_tlv_type == 20
    assert config.output.save_pointcloud is True
    assert config.gpio_reset_pin is None


def test_integer_sync_pattern_accepted(tmp_path):
    data = dict(FULL_CONFIG)
    data["packet"] = {"sync_pattern": DEFAULT_SYNC_PATTERN}
    config = load_config(write_config(tmp_path, data))
    assert config.packet.sync_pattern == DEFAULT_SYNC_PATTERN


def test_single_port_detection(tmp_path):
    data = {
        "control_serial": {"port": "/dev/serial0", "baudrate": 115200},
        "data_serial": {"port": "/dev/serial0", "baudrate": 921600},
        "gpio_reset_pin": 5,
    }
    config = load_config(write_config(tmp_path, data))
    assert config.is_single_port is True
    assert config.gpio_reset_pin == 5


def test_chirp_path_resolved_relative_to_config_file(tmp_path):
    minimal = {
        "control_serial": {"port": "COM8", "baudrate": 115200},
        "data_serial": {"port": "COM7", "baudrate": 921600},
        "chirp_config_path": "./chirp/my_config.cfg",
    }
    config = load_config(write_config(tmp_path, minimal))
    assert (
        Path(config.chirp_config_path)
        == (tmp_path / "chirp" / "my_config.cfg").resolve()
    )


def test_gui_section_defaults_and_overrides(tmp_path):
    minimal = {
        "control_serial": {"port": "COM8", "baudrate": 115200},
        "data_serial": {"port": "COM7", "baudrate": 921600},
    }
    config = load_config(write_config(tmp_path, minimal))
    assert config.gui.x_range == [-10.0, 10.0]

    minimal["gui"] = {"x_range": [-25, 25], "y_range": [0, 50]}
    config = load_config(write_config(tmp_path, minimal))
    assert config.gui.x_range == [-25, 25]
    assert config.gui.y_range == [0, 50]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("./does/not/exist.json")


def test_missing_required_section_raises(tmp_path):
    with pytest.raises(ValueError, match="data_serial"):
        load_config(
            write_config(
                tmp_path, {"control_serial": {"port": "COM8", "baudrate": 115200}}
            )
        )


def test_unknown_key_raises(tmp_path):
    data = {
        "control_serial": {"port": "COM8", "baudrate": 115200, "bogus": 1},
        "data_serial": {"port": "COM7", "baudrate": 921600},
    }
    with pytest.raises(ValueError, match="control_serial"):
        load_config(write_config(tmp_path, data))

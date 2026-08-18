# urad-mmwave

**Core Python SDK for [uRAD](https://urad.es) mmWave radar sensors by Anteral.**

*Leer en [español](README.es.md).*

This library provides the shared building blocks used by all uRAD products
based on Texas Instruments AWR/IWR mmWave chips running the out-of-box demo
firmware (mmWave SDK 3.x):

- Serial communication with the radar (dual-port USB or single-UART setups
  such as the Raspberry Pi adapter, including GPIO chip reset).
- Sending the chirp configuration exported from the TI mmWave Demo Visualizer.
- Robust TLV stream parsing: sync-word recovery, unknown-TLV skipping,
  bounded timeouts and clean shutdown (`sensorStop` is always sent on exit).
- Point cloud and temperature output in the classic uRAD text format.
- A ready-to-use command line tool: `urad-mmwave`, with an optional live
  2D point cloud viewer (`--gui`).
- Ready-made [product profiles](profiles) (serial settings, plot ranges and
  chirp configurations) for every supported product.

Supported products: **uRAD Automotive** (AWR1843AoP, 77 GHz), **uRAD
Automotive HPA** (AWR1843 ISK, 77 GHz) and **uRAD Industrial** (IWR6843AoP,
60 GHz). Firmware binaries are distributed as release assets in each
product's repository.

## Installation

Requires Python 3.9 or later.

```bash
pip install git+https://github.com/<org>/urad-mmwave-core.git
```

For Raspberry Pi single-UART setups (GPIO reset support):

```bash
pip install "urad-mmwave[rpi] @ git+https://github.com/<org>/urad-mmwave-core.git"
```

For the live point cloud viewer (`--gui`):

```bash
pip install "urad-mmwave[gui] @ git+https://github.com/<org>/urad-mmwave-core.git"
```

## Quick start (command line)

1. Connect your uRAD radar over USB and identify its two serial ports
   (control ≈ 115200 baud, data ≈ 921600 baud). On Windows check the Device
   Manager (`COMx`); on Linux they typically appear as `/dev/ttyACM0` and
   `/dev/ttyACM1`.
2. Run with your product's [profile](profiles):

```bash
urad-mmwave --config profiles/industrial/config_radar.json --data-port COM7 --control-port COM8
```

Stop with `Ctrl+C` — the sensor is stopped and the ports are released
automatically. Useful flags: `--gui` (live point cloud viewer),
`--single-port /dev/serial0` (Raspberry Pi), `--duration 10`,
`--max-frames 100`, `--no-save`, `-v`. Relative paths inside the JSON are
resolved against the profile directory, so this works from anywhere.

## Quick start (library)

```python
from urad_mmwave import PointCloudWriter, RadarSession, load_config

config = load_config("config_radar.json")

with RadarSession(config) as session, PointCloudWriter("PointCloud.txt") as out:
    for frame in session.frames():
        # frame.points is a (N, 6) numpy array: x, y, z, v, snr, noise
        print(f"Frame {frame.header.frame_number}: {len(frame.points)} points")
        out.write(frame.points, frame.timestamp)
```

## Configuration reference

| Section | Key | Default | Description |
|---|---|---|---|
| `control_serial` | `port` | — | Control/CLI UART (e.g. `COM8`, `/dev/ttyACM0`) |
| | `baudrate` | — | Usually `115200` |
| | `timeout` | `0.3` | Read timeout in seconds |
| `data_serial` | `port` | — | Data UART. Same port as control = single-UART mode |
| | `baudrate` | — | Usually `921600` |
| — | `chirp_config_path` | `./config/chirp_config.cfg` | Chirp config exported from TI mmWave Demo Visualizer |
| — | `gpio_reset_pin` | `null` | BCM pin to reset the chip (Raspberry Pi only) |
| `packet` | *(all optional)* | SDK 3.x values | Sync pattern, header formats and TLV sanity limits |
| `output` | `save_pointcloud` | `true` | Append frames to `pointcloud_path` |
| | `save_temperature` | `false` | Append TLV-9 reports to `temperature_path` |
| `display` | `print_pointcloud` | `true` | Print each detected point to stdout |
| | `print_temperature` | `false` | Print temperature reports |
| `gui` | `x_range` | `[-10, 10]` | X axis range in meters for the `--gui` viewer |
| | `y_range` | `[0, 20]` | Y axis range in meters for the `--gui` viewer |

Relative paths (`chirp_config_path`) are resolved against the directory of
the JSON file; output paths are relative to the current working directory.

### Output format

`PointCloud.txt`: one line per frame with `x y z v snr noise` repeated for
each detected object, followed by the host epoch timestamp. `Temperature.txt`:
the twelve TLV-9 fields followed by the timestamp.

## Troubleshooting

- **`SerialException: could not open port`** — wrong port name, the port is
  in use by another program (e.g. TI Visualizer), or missing permissions
  (on Linux add your user to the `dialout` group).
- **No frames arrive** — control and data ports are probably swapped, or the
  chirp configuration was not accepted; run with `-v` to see the radar's
  response to each command.
- **The radar keeps transmitting after a crash** — should not happen: the
  session always sends `sensorStop` on exit. If you killed the process hard,
  run the CLI once more or power-cycle the sensor.

## Development

```bash
git clone https://github.com/<org>/urad-mmwave-core.git
cd urad-mmwave-core
pip install -e .[dev]
pytest          # unit tests (no hardware required)
ruff check .    # lint
black .         # format
```

The parser is fully covered by hardware-independent tests built on synthetic
TLV packets — see [`tests/`](tests/).

## License

[MIT](LICENSE) © Anteral S.L.

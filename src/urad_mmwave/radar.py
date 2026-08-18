"""Radar session management: chirp configuration and serial links.

Handles the full lifecycle of a uRAD mmWave radar connection:

- Reading the chirp configuration file exported from the TI mmWave Demo
  Visualizer.
- Sending the configuration over the control UART and checking responses.
- Opening the data UART for binary streaming.
- Stopping the sensor (``sensorStop``) and releasing the ports on exit, so
  the radar does not keep transmitting after the program ends.

Supports both dual-port USB products and single-UART setups (e.g. Raspberry
Pi adapter, where control and data share one physical port and the chip can
be reset through a GPIO pin).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from time import sleep

import serial

from urad_mmwave.config import AppConfig, SerialConfig
from urad_mmwave.parser import Frame, iter_packets, read_frames

log = logging.getLogger(__name__)

SENSOR_STOP_COMMAND = "sensorStop"
_COMMAND_DELAY_S = 0.02
_GPIO_RESET_PULSE_S = 0.01
_GPIO_BOOT_WAIT_S = 1.0


def read_chirp_config(path: str | Path) -> list[str]:
    """Read a chirp configuration file and return its command lines.

    Blank lines and ``%`` comments are filtered out.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Chirp configuration file not found: {config_path}")

    commands = []
    with open(config_path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("%"):
                commands.append(line)
    return commands


def gpio_reset(pin: int) -> None:
    """Reset the radar chip through a GPIO pin (Raspberry Pi setups).

    Raises:
        ImportError: If gpiozero is not installed
            (``pip install urad-mmwave[rpi]``).
    """
    try:
        from gpiozero import OutputDevice
    except ImportError as exc:
        raise ImportError(
            "gpiozero is required for GPIO reset. "
            "Install it with: pip install urad-mmwave[rpi]"
        ) from exc

    reset_pin = OutputDevice(pin)
    reset_pin.off()
    sleep(_GPIO_RESET_PULSE_S)
    reset_pin.on()
    sleep(_GPIO_BOOT_WAIT_S)


def _open_port(config: SerialConfig) -> serial.Serial:
    return serial.Serial(
        config.port,
        config.baudrate,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=config.timeout,
    )


def _send_command(port: serial.Serial, command: str) -> str:
    """Send one command line and return the radar's echoed response."""
    port.write((command.strip() + "\n").encode("ascii"))
    sleep(_COMMAND_DELAY_S)
    response = bytearray()
    while port.in_waiting > 0:
        response += port.read(port.in_waiting)
    return response.decode(errors="replace").strip()


class RadarSession:
    """Context manager owning the serial link(s) to a uRAD mmWave radar.

    On entry: optionally resets the chip via GPIO, sends the chirp
    configuration over the control port and opens the data port.
    On exit: closes the data port, sends ``sensorStop`` and releases
    everything, even if an exception (including Ctrl+C) occurred.

    Example:
        config = load_config("config_radar.json")
        with RadarSession(config) as session:
            for frame in session.frames():
                ...
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._data_port: serial.Serial | None = None
        self._stopped = False

    def __enter__(self) -> RadarSession:
        config = self._config
        if config.gpio_reset_pin is not None:
            log.info("Resetting radar via GPIO pin %d", config.gpio_reset_pin)
            gpio_reset(config.gpio_reset_pin)

        commands = read_chirp_config(config.chirp_config_path)
        log.info(
            "Configuring radar on %s at %d baud (%d commands)",
            config.control_serial.port,
            config.control_serial.baudrate,
            len(commands),
        )
        self._configure(commands)

        log.info(
            "Opening data port %s at %d baud",
            config.data_serial.port,
            config.data_serial.baudrate,
        )
        self._data_port = _open_port(config.data_serial)
        self._data_port.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return False

    def _configure(self, commands: list[str]) -> None:
        with _open_port(self._config.control_serial) as control_port:
            control_port.reset_input_buffer()
            for command in commands:
                response = _send_command(control_port, command)
                log.debug("%s -> %s", command, response or "<no response>")
                if "Error" in response:
                    log.warning(
                        "Radar reported an error for '%s': %s", command, response
                    )

    def frames(self) -> Iterator[Frame]:
        """Yield decoded out-of-box frames from the data port."""
        if self._data_port is None:
            raise RuntimeError("Session not started; use 'with RadarSession(...)'")
        return read_frames(self._data_port, self._config.packet)

    def packets(self) -> Iterator[tuple[tuple, bytes, float]]:
        """Yield raw ``(header_fields, payload, timestamp)`` packets.

        For application firmwares (e.g. 3D people counting) that share the
        packet framing but use their own TLV set — decode the payload with
        the application's parser.
        """
        if self._data_port is None:
            raise RuntimeError("Session not started; use 'with RadarSession(...)'")
        return iter_packets(
            self._data_port,
            self._config.packet.sync_pattern,
            self._config.packet.header_format,
        )

    def stop(self) -> None:
        """Stop the sensor and release the serial ports (idempotent)."""
        if self._stopped:
            return
        self._stopped = True

        if self._data_port is not None and self._data_port.is_open:
            self._data_port.close()

        try:
            with _open_port(self._config.control_serial) as control_port:
                _send_command(control_port, SENSOR_STOP_COMMAND)
            log.info("Sensor stopped")
        except serial.SerialException as exc:
            log.warning("Could not send %s: %s", SENSOR_STOP_COMMAND, exc)

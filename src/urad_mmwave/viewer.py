"""Live 2D point cloud viewer for the out-of-box demo (optional GUI extra).

Frames are read from the radar in a background thread; the Qt main thread
draws only the most recent frame, so a slow display never backs up the
serial stream. Marker size encodes the SNR of each detected point.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator

import numpy as np

from urad_mmwave.config import AppConfig
from urad_mmwave.parser import Frame

log = logging.getLogger(__name__)

_MIN_POINT_SIZE = 4.0
_MAX_POINT_SIZE = 16.0
_REFRESH_INTERVAL_MS = 33


def point_sizes(snr: np.ndarray) -> np.ndarray:
    """Map SNR values (0.1 dB units) to marker sizes in pixels."""
    return np.clip(_MIN_POINT_SIZE + snr / 40.0, _MIN_POINT_SIZE, _MAX_POINT_SIZE)


def _import_pyqtgraph():
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore
    except ImportError as exc:
        raise ImportError(
            "The live viewer requires pyqtgraph and a Qt binding. "
            "Install them with: pip install urad-mmwave[gui]"
        ) from exc
    return pg, QtCore


def run_viewer(
    config: AppConfig,
    frames: Iterator[Frame],
    on_frame: Callable[[Frame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        config: Application configuration (``config.gui`` sets the axis
            ranges of the plot).
        frames: Iterator of decoded frames, e.g. ``RadarSession.frames()``.
        on_frame: Optional callback invoked for every frame from the reader
            thread (used by the CLI for printing and file output).

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()

    latest: deque = deque(maxlen=1)
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in frames:
                if on_frame is not None:
                    on_frame(frame)
                latest.append(frame)
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD out-of-box demo")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD out-of-box demo")
    plot = window.addPlot(title="Point cloud (top view)")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    plot.setXRange(config.gui.x_range[0], config.gui.x_range[1], padding=0)
    plot.setYRange(config.gui.y_range[0], config.gui.y_range[1], padding=0)
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(20, 20, 20, 200))
    plot.addItem(scatter)

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return
        if len(frame.points):
            scatter.setData(
                x=frame.points[:, 0],
                y=frame.points[:, 1],
                size=point_sizes(frame.points[:, 4]),
            )
        else:
            scatter.setData(x=[], y=[])
        window.setWindowTitle(
            f"uRAD out-of-box demo — frame {frame.header.frame_number} "
            f"({len(frame.points)} points)"
        )

    reader.start()
    timer = QtCore.QTimer()
    timer.timeout.connect(_refresh)
    timer.start(_REFRESH_INTERVAL_MS)

    log.info("Viewer running; close the window to stop")
    if hasattr(app, "exec"):
        app.exec()
    else:  # older Qt bindings
        app.exec_()
    stop_reading.set()
    # Wait for the reader to leave the serial read before the caller closes
    # the ports underneath it (pyserial raises from a closed port on Windows).
    reader.join(timeout=2.0)

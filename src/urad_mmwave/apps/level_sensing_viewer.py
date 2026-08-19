"""Live views for the level sensing application (gui extra).

Two windows are available: the distance-over-time view
(:func:`run_viewer`) plots the three high-accuracy ranges as scrolling
curves, and the spectrum view (:func:`run_spectrum_viewer`) shows the
full range profile computed from the raw ADC samples, with the detected
peaks marked — the modern equivalent of the legacy ``plotSpectrum``
GUI. Frames are read from the radar in a background thread; the Qt main
thread draws only the most recent data, so a slow display never backs
up the serial stream.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterator

import numpy as np

from urad_mmwave.apps.level_sensing import (
    LevelSensingSettings,
    range_spectrum,
)
from urad_mmwave.viewer import _import_pyqtgraph

log = logging.getLogger(__name__)

_SPECTRUM_FLOOR_DB = -80.0

_REFRESH_INTERVAL_MS = 100
_HISTORY_SECONDS = 10.0

# One distinguishable color per range curve.
_RANGE_COLORS = [(230, 25, 75), (60, 180, 75), (67, 99, 216)]


def run_viewer(samples: Iterator) -> None:
    """Open the distance-over-time window and stream until it is closed.

    Args:
        samples: Iterator of :class:`~urad_mmwave.apps.level_sensing.
            LevelSensingFrame`, e.g.
            :func:`urad_mmwave.apps.level_sensing.stream`.

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()

    history: deque = deque(maxlen=4096)  # (timestamp, r1, r2, r3)
    lock = threading.Lock()
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in samples:
                with lock:
                    history.append((frame.timestamp, *frame.ranges))
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD level sensing")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD level sensing")
    plot = window.addPlot(title="High accuracy ranges over time")
    plot.setLabel("bottom", "Time (s)")
    plot.setLabel("left", "Distance (m)")
    plot.showGrid(x=True, y=True)
    plot.addLegend()

    curves = [
        plot.plot(pen=pg.mkPen(color=color, width=2), name=f"Range {i + 1}")
        for i, color in enumerate(_RANGE_COLORS)
    ]

    def _refresh() -> None:
        with lock:
            if not history:
                return
            data = list(history)
        start = data[0][0]
        newest = data[-1][0]
        elapsed = [row[0] - start for row in data]
        first = 0
        while newest - data[first][0] > _HISTORY_SECONDS:
            first += 1
        for i, curve in enumerate(curves):
            curve.setData(elapsed[first:], [row[1 + i] for row in data[first:]])
        window.setWindowTitle(
            "uRAD level sensing — "
            + "  |  ".join(f"r{i + 1}: {data[-1][1 + i]:.4f} m" for i in range(3))
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
    # Wait for the reader to leave the serial read before the generator's
    # cleanup closes the ports underneath it (pyserial raises on Windows).
    reader.join(timeout=2.0)


def run_spectrum_viewer(settings: LevelSensingSettings, samples: Iterator) -> None:
    """Open the range-profile window and stream until it is closed.

    Args:
        settings: The measurement settings used to configure the radar
            (``raw_iq`` must be set so the firmware streams the ADC
            samples; the plot spans ``range_min``..``range_max``).
        samples: Iterator of :class:`~urad_mmwave.apps.level_sensing.
            LevelSensingFrame` with ``iq`` populated, e.g.
            :func:`urad_mmwave.apps.level_sensing.stream`.

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()

    latest: deque = deque(maxlen=1)
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in samples:
                latest.append(frame)
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD level sensing")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD level sensing")
    plot = window.addPlot(title="Range profile")
    plot.setLabel("bottom", "Distance (m)")
    plot.setLabel("left", "Power (dB)")
    plot.showGrid(x=True, y=True)
    plot.setXRange(max(settings.range_min, 0.0), settings.range_max, padding=0)
    plot.setYRange(_SPECTRUM_FLOOR_DB, 0.0, padding=0)

    curve = plot.plot(pen=pg.mkPen((0, 0, 255), width=2))
    peaks_scatter = pg.ScatterPlotItem(
        pen=pg.mkPen("k", width=1), brush=pg.mkBrush(230, 25, 75, 200), size=10
    )
    plot.addItem(peaks_scatter)

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return
        if frame.iq is None or not len(frame.iq):
            return

        range_axis, power_db = range_spectrum(frame.iq, settings.maximum_distance)
        curve.setData(range_axis, power_db)

        spots = []
        titles = []
        for i, measured in enumerate(frame.ranges):
            index = int(np.argmin(np.abs(range_axis - measured)))
            spots.append({"pos": (measured, power_db[index])})
            titles.append(f"r{i + 1}: {measured:.4f} m ({power_db[index]:.1f} dB)")
        peaks_scatter.setData(spots)
        window.setWindowTitle("uRAD level sensing — " + "  |  ".join(titles))

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
    # Wait for the reader to leave the serial read before the generator's
    # cleanup closes the ports underneath it (pyserial raises on Windows).
    reader.join(timeout=2.0)

"""Live distance-over-time view for the level sensing application (gui extra).

Plots the three high-accuracy range measurements as scrolling curves and
shows the latest values in the window title. Frames are read from the
radar in a background thread; the Qt main thread draws only the samples
received so far, so a slow display never backs up the serial stream.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterator

from urad_mmwave.viewer import _import_pyqtgraph

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 100
_HISTORY_SECONDS = 120.0

# One distinguishable color per range curve.
_RANGE_COLORS = [(230, 25, 75), (60, 180, 75), (67, 99, 216)]


def run_viewer(samples: Iterator[tuple[float, tuple[float, float, float]]]) -> None:
    """Open the viewer window and stream samples until it is closed.

    Args:
        samples: Iterator of ``(timestamp, (r1, r2, r3))`` tuples, e.g.
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
            for timestamp, ranges in samples:
                with lock:
                    history.append((timestamp, *ranges))
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

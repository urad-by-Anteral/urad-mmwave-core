"""Live views for the Automated Parking application (optional GUI extra).

Three panels, refreshed with the latest packet of each subframe:

- **Long range 2D (0-50 m)**: top view of the long range point cloud.
- **Short range 3D (0-10 m)**: top view of the short range point cloud,
  colored by height (z), with the nearest obstacle marked.
- **Range profiles**: zero-Doppler range profile of both subframes, in dB.

Same threading model as the other viewers: packets are read in a background
thread and the Qt timer draws the most recent frame of each subframe.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np

from urad_mmwave.apps.automated_parking import (
    SUBFRAME_LONG_RANGE,
    SUBFRAME_SHORT_RANGE,
    ParkingFrame,
    nearest_obstacle,
)
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
LONG_RANGE_M = 50.0
SHORT_RANGE_M = 10.0
HEIGHT_LIMITS_M = (-1.0, 2.0)
"""Height range mapped onto the short range color scale."""


def height_colors(z: np.ndarray) -> np.ndarray:
    """RGBA colors (0-255) for heights: blue (low) → green → red (high)."""
    lo, hi = HEIGHT_LIMITS_M
    t = np.clip((np.asarray(z, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(2 * t - 1, 0, 1)
    b = np.clip(1 - 2 * t, 0, 1)
    g = 1 - r - b
    rgba = np.column_stack([r, g, b, np.full_like(t, 0.85)])
    return (rgba * 255).astype(int)


def _wedge(
    radius: float, half_angle_deg: float = 75.0
) -> tuple[np.ndarray, np.ndarray]:
    a = np.radians(np.linspace(-half_angle_deg, half_angle_deg, 90))
    xs = np.concatenate([[0.0], radius * np.sin(a), [0.0]])
    ys = np.concatenate([[0.0], radius * np.cos(a), [0.0]])
    return xs, ys


def run_viewer(
    frames: Iterator[ParkingFrame],
    on_frame: Callable[[ParkingFrame], None] | None = None,
    range_bin_m: dict[int, float] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        frames: Iterator of :class:`ParkingFrame` (both subframes interleaved).
        on_frame: Optional callback for every frame, from the reader thread.
        range_bin_m: Meters per range-profile bin for each subframe (see
            :func:`~urad_mmwave.apps.automated_parking.range_bin_sizes`);
            without it the profiles are plotted against the bin index.
    """
    pg, QtCore = _import_pyqtgraph()
    range_bin_m = range_bin_m or {}

    latest: dict[int, ParkingFrame] = {}
    lock = threading.Lock()
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in frames:
                if on_frame is not None:
                    on_frame(frame)
                with lock:
                    latest[frame.subframe] = frame
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOption("antialias", True)
    title = "uRAD Automated Parking"
    app = pg.mkQApp(title)
    window = pg.GraphicsLayoutWidget(show=True, title=title)
    window.resize(1100, 800)
    outline_pen = pg.mkPen((40, 150, 70), width=1)

    far = window.addPlot(row=0, col=0, title="Long range 2D (0-50 m)")
    far.setLabel("bottom", "X (m)")
    far.setLabel("left", "Y (m)")
    far.setAspectLocked(True)
    far.showGrid(x=True, y=True, alpha=0.3)
    far.setXRange(-LONG_RANGE_M, LONG_RANGE_M, padding=0.02)
    far.setYRange(0, LONG_RANGE_M, padding=0.02)
    far.plot(*_wedge(LONG_RANGE_M), pen=outline_pen)
    far.plot(
        *_wedge(SHORT_RANGE_M),
        pen=pg.mkPen((200, 120, 0), width=1, style=QtCore.Qt.PenStyle.DashLine),
    )
    far_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 110, 190, 200))
    far.addItem(far_scatter)

    near = window.addPlot(row=0, col=1, title="Short range 3D (0-10 m), color = height")
    near.setLabel("bottom", "X (m)")
    near.setLabel("left", "Y (m)")
    near.setAspectLocked(True)
    near.showGrid(x=True, y=True, alpha=0.3)
    near.setXRange(-SHORT_RANGE_M, SHORT_RANGE_M, padding=0.02)
    near.setYRange(0, SHORT_RANGE_M, padding=0.02)
    near.plot(*_wedge(SHORT_RANGE_M), pen=outline_pen)
    near_scatter = pg.ScatterPlotItem(pen=None)
    near.addItem(near_scatter)
    nearest_ring = pg.PlotCurveItem(pen=pg.mkPen((220, 40, 40), width=2))
    near.addItem(nearest_ring)
    nearest_text = pg.TextItem("", color=(220, 40, 40), anchor=(0.5, 1.0))
    near.addItem(nearest_text)

    prof = window.addPlot(row=1, col=0, colspan=2, title="Range profile (zero Doppler)")
    prof.setLabel("left", "Power (dB, relative)")
    prof.setLabel("bottom", "Range (m)" if range_bin_m else "Range bin")
    prof.addLegend(offset=(-10, 10))
    prof.showGrid(x=True, y=True, alpha=0.3)
    curves = {
        SUBFRAME_LONG_RANGE: prof.plot(
            pen=pg.mkPen((0, 110, 190), width=1), name="2D 50 m"
        ),
        SUBFRAME_SHORT_RANGE: prof.plot(
            pen=pg.mkPen((200, 120, 0), width=1), name="3D 10 m"
        ),
    }
    window.ci.layout.setRowStretchFactor(0, 3)
    window.ci.layout.setRowStretchFactor(1, 1)

    arc = np.radians(np.linspace(-75, 75, 90))

    def _refresh() -> None:
        with lock:
            frames_now = dict(latest)
        f = frames_now.get(SUBFRAME_LONG_RANGE)
        if f is not None:
            p = f.points
            if len(p):
                far_scatter.setData(x=p[:, 0], y=p[:, 1], size=point_sizes(p[:, 4]))
            else:
                far_scatter.setData(x=[], y=[])
        n = frames_now.get(SUBFRAME_SHORT_RANGE)
        if n is not None:
            p = n.points
            if len(p):
                near_scatter.setData(
                    x=p[:, 0],
                    y=p[:, 1],
                    size=point_sizes(p[:, 4]),
                    brush=[pg.mkBrush(*c) for c in height_colors(p[:, 2])],
                )
            else:
                near_scatter.setData(x=[], y=[])
            d = nearest_obstacle(p)
            if d is None:
                nearest_ring.setData([], [])
                nearest_text.setText("")
            else:
                nearest_ring.setData(d * np.sin(arc), d * np.cos(arc))
                nearest_text.setText(f"{d:.2f} m")
                nearest_text.setPos(0, d)
        for sub, frame in frames_now.items():
            curve = curves.get(sub)
            if curve is None or not len(frame.range_profile_db):
                continue
            step = range_bin_m.get(sub, 1.0)
            x = np.arange(len(frame.range_profile_db)) * step
            curve.setData(x, frame.range_profile_db)
        counts = ", ".join(
            f"{frames_now[s].name}: {len(frames_now[s].points)}"
            for s in sorted(frames_now)
        )
        window.setWindowTitle(f"{title} — {counts}")

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
    # Let the reader leave the serial read before the caller closes the ports.
    reader.join(timeout=2.0)

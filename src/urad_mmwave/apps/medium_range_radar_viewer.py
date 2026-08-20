"""Live 2D top-view for the Medium Range Radar application (gui extra).

Shows both subframes on one X-Y plot, following the color scheme of the
TI MATLAB visualizer:

- MRR subframe: detected points in green, tracked objects as orange
  squares with a velocity vector and a dashed extent box.
- USRR subframe: detected points in blue, DBSCAN clusters as gray
  rectangles, parking assist border as a red arc (bins at the maximum
  parking range are free space).

MRR and USRR frames alternate on the stream, so the viewer keeps the most
recent frame of each subframe and redraws both. Frames are read from the
radar in a background thread; the Qt main thread draws only the most
recent frames, so a slow display never backs up the serial stream.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np

from urad_mmwave.apps.medium_range_radar import (
    SUBFRAME_MRR,
    SUBFRAME_USRR,
    MrrFrame,
    parking_assist_to_xy,
)
from urad_mmwave.viewer import _import_pyqtgraph

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_DEFAULT_X_RANGE = (-50.0, 50.0)
_DEFAULT_Y_RANGE = (0.0, 130.0)

_MRR_POINT_COLOR = (44, 160, 44)  # green
_USRR_POINT_COLOR = (31, 119, 180)  # blue
_CLUSTER_COLOR = (120, 120, 120)  # gray
_TRACKER_COLOR = (255, 130, 48)  # orange
_PARKING_COLOR = (214, 39, 40)  # red

_VELOCITY_SCALE_S = 1.0  # vector length = distance covered in one second


def _rectangles_xy(
    rects: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Connected x/y arrays drawing one closed box per (cx, cy, sx, sy).

    The boxes are separated by NaN vertices so a single plot item can draw
    all of them; sizes are half-extents around the center.
    """
    x: list[float] = []
    y: list[float] = []
    for cx, cy, sx, sy in rects:
        x += [cx - sx, cx + sx, cx + sx, cx - sx, cx - sx, np.nan]
        y += [cy - sy, cy - sy, cy + sy, cy + sy, cy - sy, np.nan]
    return np.array(x), np.array(y)


def _segments_xy(
    segments: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Connected x/y arrays drawing one line per (x0, y0, x1, y1)."""
    x: list[float] = []
    y: list[float] = []
    for x0, y0, x1, y1 in segments:
        x += [x0, x1, np.nan]
        y += [y0, y1, np.nan]
    return np.array(x), np.array(y)


def run_viewer(
    frames: Iterator[MrrFrame],
    on_frame: Callable[[MrrFrame], None] | None = None,
    x_range: tuple[float, float] = _DEFAULT_X_RANGE,
    y_range: tuple[float, float] = _DEFAULT_Y_RANGE,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        frames: Iterator of decoded frames (both subframes interleaved).
        on_frame: Optional callback invoked for every frame from the reader
            thread (used by the CLI for printing and file output).
        x_range: Initial X axis range in meters.
        y_range: Initial Y axis range in meters.

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()

    latest: dict[int, MrrFrame] = {}
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in frames:
                if on_frame is not None:
                    on_frame(frame)
                latest[frame.subframe] = frame
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD medium range radar")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD medium range radar")
    plot = window.addPlot(title="Medium Range Radar (top view)")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    plot.addLegend()
    plot.setXRange(x_range[0], x_range[1], padding=0)
    plot.setYRange(y_range[0], y_range[1], padding=0)

    # Qt6 moved the pen styles into the PenStyle enum; Qt5 has them flat.
    pen_styles = getattr(QtCore.Qt, "PenStyle", QtCore.Qt)

    mrr_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(*_MRR_POINT_COLOR, 160), size=6, name="MRR points"
    )
    plot.addItem(mrr_scatter)
    usrr_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(*_USRR_POINT_COLOR, 160), size=6, name="USRR points"
    )
    plot.addItem(usrr_scatter)
    cluster_curve = plot.plot(
        [], [], pen=pg.mkPen(color=_CLUSTER_COLOR, width=2), name="Clusters"
    )
    tracker_scatter = pg.ScatterPlotItem(
        pen=pg.mkPen("k", width=1),
        brush=pg.mkBrush(*_TRACKER_COLOR, 200),
        symbol="s",
        size=14,
        name="Tracked objects",
    )
    plot.addItem(tracker_scatter)
    tracker_boxes = plot.plot(
        [],
        [],
        pen=pg.mkPen(
            color=_TRACKER_COLOR, width=1, style=getattr(pen_styles, "DashLine")
        ),
    )
    velocity_curve = plot.plot([], [], pen=pg.mkPen(color=_TRACKER_COLOR, width=2))
    parking_curve = plot.plot(
        [], [], pen=pg.mkPen(color=_PARKING_COLOR, width=2), name="Parking border"
    )

    def _refresh() -> None:
        mrr = latest.get(SUBFRAME_MRR)
        usrr = latest.get(SUBFRAME_USRR)

        if mrr is not None:
            if len(mrr.points):
                mrr_scatter.setData(x=mrr.points[:, 0], y=mrr.points[:, 1])
            else:
                mrr_scatter.setData(x=[], y=[])

            tracker_scatter.setData(
                x=[t.position[0] for t in mrr.trackers],
                y=[t.position[1] for t in mrr.trackers],
            )
            box_x, box_y = _rectangles_xy(
                [(*t.position, *t.size) for t in mrr.trackers]
            )
            tracker_boxes.setData(box_x, box_y)
            vel_x, vel_y = _segments_xy(
                [
                    (
                        t.position[0],
                        t.position[1],
                        t.position[0] + t.velocity[0] * _VELOCITY_SCALE_S,
                        t.position[1] + t.velocity[1] * _VELOCITY_SCALE_S,
                    )
                    for t in mrr.trackers
                ]
            )
            velocity_curve.setData(vel_x, vel_y)

        if usrr is not None:
            if len(usrr.points):
                usrr_scatter.setData(x=usrr.points[:, 0], y=usrr.points[:, 1])
            else:
                usrr_scatter.setData(x=[], y=[])

            cluster_x, cluster_y = _rectangles_xy(
                [tuple(c) for c in usrr.clusters]
            )
            cluster_curve.setData(cluster_x, cluster_y)

            park_x, park_y = parking_assist_to_xy(usrr.parking_assist)
            parking_curve.setData(park_x, park_y)

        title = "uRAD medium range radar"
        if mrr is not None:
            title += (
                f" — MRR: {len(mrr.points)} points, {len(mrr.trackers)} tracked"
            )
        if usrr is not None:
            title += f" — USRR: {len(usrr.points)} points, {len(usrr.clusters)} clusters"
        window.setWindowTitle(title)

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

"""Live 2D top-view for the 3D People Tracking application (gui extra).

Shows the detected point cloud (marker size encodes SNR), the tracked
people as numbered circles colored per track id, and the tracker zones
defined in the chirp configuration (``boundaryBox``,
``staticBoundaryBox`` and ``presenceBoundaryBox``).

Frames are read from the radar in a background thread; the Qt main thread
draws only the most recent frame, so a slow display never backs up the
serial stream.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator

from urad_mmwave.apps.people_tracking import (
    PeopleTrackingFrame,
    points_to_xy,
    read_boundary_boxes,
)
from urad_mmwave.config import AppConfig
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ZONE_MARGIN_M = 1.0

# Zone name -> (legend label, RGB pen color, Qt pen style name).
_ZONE_STYLES = {
    "boundaryBox": ("Boundary box", (120, 120, 120), "DashLine"),
    "staticBoundaryBox": ("Static boundary box", (31, 119, 180), "DashLine"),
    "presenceBoundaryBox": ("Presence boundary box", (44, 160, 44), "DotLine"),
}

# Distinguishable track colors, indexed by track id modulo the palette size.
_TRACK_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 130, 48),
    (67, 99, 216),
    (145, 30, 180),
    (66, 212, 244),
    (240, 50, 230),
    (128, 128, 0),
]


def track_color(tid: int) -> tuple[int, int, int]:
    """Stable display color for a track id."""
    return _TRACK_COLORS[tid % len(_TRACK_COLORS)]


def run_viewer(
    config: AppConfig,
    frames: Iterator[PeopleTrackingFrame],
    on_frame: Callable[[PeopleTrackingFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        config: Application configuration; the tracker zones are read from
            ``config.chirp_config_path`` and the plot ranges derive from the
            ``boundaryBox`` zone (falling back to ``config.gui``).
        frames: Iterator of decoded frames.
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

    zones = read_boundary_boxes(config.chirp_config_path)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD people tracking")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD people tracking")
    plot = window.addPlot(title="People tracking (top view)")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    plot.addLegend()

    # Plot ranges: frame the boundary box if the config defines one.
    if "boundaryBox" in zones:
        xmin, xmax, ymin, ymax = zones["boundaryBox"][:4]
        plot.setXRange(xmin - _ZONE_MARGIN_M, xmax + _ZONE_MARGIN_M, padding=0)
        plot.setYRange(min(ymin, 0.0), ymax + _ZONE_MARGIN_M, padding=0)
    else:
        plot.setXRange(config.gui.x_range[0], config.gui.x_range[1], padding=0)
        plot.setYRange(config.gui.y_range[0], config.gui.y_range[1], padding=0)

    # Qt6 moved the pen styles into the PenStyle enum; Qt5 has them flat.
    pen_styles = getattr(QtCore.Qt, "PenStyle", QtCore.Qt)
    for name, box in zones.items():
        label, color, pen_style = _ZONE_STYLES.get(name, (name, (0, 0, 0), "DashLine"))
        xmin, xmax, ymin, ymax = box[:4]
        pen = pg.mkPen(color=color, width=2, style=getattr(pen_styles, pen_style))
        plot.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            pen=pen,
            name=label,
        )

    points_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(20, 20, 20, 140), name="Points"
    )
    plot.addItem(points_scatter)
    targets_scatter = pg.ScatterPlotItem(pen=pg.mkPen("k", width=1))
    plot.addItem(targets_scatter)
    track_labels: dict[int, object] = {}

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return

        if len(frame.points):
            x, y = points_to_xy(frame.points)
            points_scatter.setData(x=x, y=y, size=point_sizes(frame.points[:, 4]))
        else:
            points_scatter.setData(x=[], y=[])

        spots = [
            {
                "pos": (target.position[0], target.position[1]),
                "size": 22,
                "brush": pg.mkBrush(*track_color(target.tid), 180),
            }
            for target in frame.targets
        ]
        targets_scatter.setData(spots)

        seen = set()
        for target in frame.targets:
            seen.add(target.tid)
            label = track_labels.get(target.tid)
            if label is None:
                label = pg.TextItem(str(target.tid), color="k", anchor=(0.5, 0.5))
                plot.addItem(label)
                track_labels[target.tid] = label
            label.setPos(target.position[0], target.position[1])
        for tid in list(track_labels):
            if tid not in seen:
                plot.removeItem(track_labels.pop(tid))

        presence = ""
        if frame.presence is not None:
            presence = f", presence {'ON' if frame.presence else 'off'}"
        window.setWindowTitle(
            f"uRAD people tracking — {len(frame.targets)} tracked, "
            f"{len(frame.points)} points{presence}"
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

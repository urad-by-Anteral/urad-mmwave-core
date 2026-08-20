"""Live 2D top-view for the Area Scanner application (gui extra).

Shows the dynamic point cloud (marker size encodes SNR), the static point
cloud (newly added static objects) as magenta squares, the tracked objects
as numbered circles colored per track id with a projection line colored by
the zone of the projected position (red = critical, orange = warning,
green = clear), the radial occupancy zones as arcs around the sensor and
the tracker zones defined in the chirp configuration (``boundaryBox`` and
``staticBoundaryBox``).

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

import numpy as np

from urad_mmwave.apps.area_scanner import (
    AreaScannerFrame,
    ZoneConfig,
    classify_zones,
    points_to_xy,
    projected_position,
    read_boundary_boxes,
    zone_of,
)
from urad_mmwave.apps.people_tracking_viewer import track_color
from urad_mmwave.config import AppConfig
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ZONE_MARGIN_M = 1.0
_ARC_POINTS = 91  # half circle in 2-degree steps

# Zone name -> (legend label, RGB pen color, Qt pen style name).
_BOX_STYLES = {
    "boundaryBox": ("Boundary box", (120, 120, 120), "DashLine"),
    "staticBoundaryBox": ("Static boundary box", (31, 119, 180), "DashLine"),
}

# Projection line colors per zone of the projected position.
_PROJECTION_COLORS = {
    "critical": (214, 39, 40),
    "warning": (255, 165, 0),
    "clear": (44, 160, 44),
}


def _zone_arc(radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Half-circle arc (y >= 0) of the given radius, centered on the sensor."""
    theta = np.linspace(0.0, np.pi, _ARC_POINTS)
    return radius * np.cos(theta), radius * np.sin(theta)


def run_viewer(
    config: AppConfig,
    frames: Iterator[AreaScannerFrame],
    zones: ZoneConfig,
    on_frame: Callable[[AreaScannerFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        config: Application configuration; the tracker boxes are read from
            ``config.chirp_config_path`` and the plot ranges derive from the
            ``boundaryBox`` zone (falling back to the warning zone radius).
        frames: Iterator of decoded frames.
        zones: Radial occupancy zones drawn as arcs and used for the zone
            status in the window title.
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

    boxes = read_boundary_boxes(config.chirp_config_path)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD area scanner")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD area scanner")
    plot = window.addPlot(title="Area scanner (top view)")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    plot.addLegend()

    # Plot ranges: frame the boundary box if the config defines one.
    if "boundaryBox" in boxes:
        xmin, xmax, ymin, ymax = boxes["boundaryBox"][:4]
        plot.setXRange(xmin - _ZONE_MARGIN_M, xmax + _ZONE_MARGIN_M, padding=0)
        plot.setYRange(min(ymin, 0.0), ymax + _ZONE_MARGIN_M, padding=0)
    else:
        reach = zones.warning[1] + _ZONE_MARGIN_M
        plot.setXRange(-reach, reach, padding=0)
        plot.setYRange(0.0, reach, padding=0)

    # Qt6 moved the pen styles into the PenStyle enum; Qt5 has them flat.
    pen_styles = getattr(QtCore.Qt, "PenStyle", QtCore.Qt)
    for name, box in boxes.items():
        label, color, pen_style = _BOX_STYLES.get(name, (name, (0, 0, 0), "DashLine"))
        xmin, xmax, ymin, ymax = box[:4]
        pen = pg.mkPen(color=color, width=2, style=getattr(pen_styles, pen_style))
        plot.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            pen=pen,
            name=label,
        )

    # Occupancy zone arcs around the sensor (start arcs only when nonzero).
    zone_arcs = (
        (zones.critical[0], _PROJECTION_COLORS["critical"], None),
        (zones.critical[1], _PROJECTION_COLORS["critical"], "Critical zone"),
        (zones.warning[0], _PROJECTION_COLORS["warning"], None),
        (zones.warning[1], _PROJECTION_COLORS["warning"], "Warning zone"),
    )
    drawn_radii = set()
    for radius, color, label in zone_arcs:
        if radius <= 0 or radius in drawn_radii:
            continue
        drawn_radii.add(radius)
        x, y = _zone_arc(radius)
        pen = pg.mkPen(color=color, width=2, style=pen_styles.DashLine)
        plot.plot(x, y, pen=pen, name=label)

    dynamic_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(0, 130, 170, 160), name="Dynamic points"
    )
    plot.addItem(dynamic_scatter)
    static_scatter = pg.ScatterPlotItem(
        pen=None,
        brush=pg.mkBrush(200, 30, 160, 200),
        symbol="s",
        size=9,
        name="Static points",
    )
    plot.addItem(static_scatter)
    targets_scatter = pg.ScatterPlotItem(pen=pg.mkPen("k", width=1))
    plot.addItem(targets_scatter)
    track_labels: dict[int, object] = {}
    projection_curves = {
        name: plot.plot([], [], pen=pg.mkPen(color=color, width=2), connect="pairs")
        for name, color in _PROJECTION_COLORS.items()
    }

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return

        if len(frame.dynamic_points):
            x, y = points_to_xy(frame.dynamic_points)
            dynamic_scatter.setData(
                x=x, y=y, size=point_sizes(frame.dynamic_points[:, 4])
            )
        else:
            dynamic_scatter.setData(x=[], y=[])

        if len(frame.static_points):
            static_scatter.setData(
                x=frame.static_points[:, 0], y=frame.static_points[:, 1]
            )
        else:
            static_scatter.setData(x=[], y=[])

        spots = [
            {
                "pos": (target.position[0], target.position[1]),
                "size": 22,
                "brush": pg.mkBrush(*track_color(target.tid), 180),
            }
            for target in frame.targets
        ]
        targets_scatter.setData(spots)

        # Projection lines, grouped per zone color and drawn as point pairs.
        segments: dict[str, list[float]] = {name: [] for name in _PROJECTION_COLORS}
        seen = set()
        for target in frame.targets:
            seen.add(target.tid)
            label = track_labels.get(target.tid)
            if label is None:
                label = pg.TextItem(str(target.tid), color="k", anchor=(0.5, 0.5))
                plot.addItem(label)
                track_labels[target.tid] = label
            label.setPos(target.position[0], target.position[1])

            projected = projected_position(target, zones.projection_time)
            name = zone_of(float(np.linalg.norm(projected)), zones)
            segments[name] += [
                (target.position[0], target.position[1]),
                (projected[0], projected[1]),
            ]
        for tid in list(track_labels):
            if tid not in seen:
                plot.removeItem(track_labels.pop(tid))
        for name, curve in projection_curves.items():
            if segments[name]:
                xy = np.array(segments[name])
                curve.setData(xy[:, 0], xy[:, 1], connect="pairs")
            else:
                curve.setData([], [])

        status = classify_zones(frame, zones)
        window.setWindowTitle(
            f"uRAD area scanner — {len(frame.dynamic_points)} dynamic, "
            f"{len(frame.static_points)} static, {len(frame.targets)} tracks, "
            f"zone {status.label}"
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

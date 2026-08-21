"""Live 2D views for the Small Obstacle Detection application (gui extra).

Top view (X-Y) with the detected point cloud where marker color encodes
height — near-ground points glow warm (red/orange) so low obstacles stand
out against taller clutter — plus the tracked obstacles as numbered
circles, the occupancy zones from the chirp configuration (solid red while
occupied, green outline while clear) and the tracker boundary box. A side
view (Y-Z) below shows the same points against the zone height limits, to
make near-ground obstacles easy to read.

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

from urad_mmwave.apps.people_tracking import read_boundary_boxes
from urad_mmwave.apps.small_obstacle import (
    SmallObstacleFrame,
    points_to_xyz,
    read_zones,
)
from urad_mmwave.config import AppConfig
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ZONE_MARGIN_M = 0.5

# Default height color range (m, radar-relative). With the default configs
# the sensor sits ~0.5 m above the ground, so ground level is around -0.5 m.
_HEIGHT_RANGE_M = (-0.5, 1.5)

# Height gradient stops (low -> high): warm red near the ground through
# orange to a cool blue overhead, so low obstacles stand out.
_HEIGHT_GRADIENT = (
    (214, 39, 40),  # near ground
    (255, 152, 0),
    (120, 170, 90),
    (31, 119, 180),  # high
)

# Distinguishable obstacle track colors, indexed by track id modulo the
# palette size.
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

_ZONE_CLEAR_COLOR = (44, 160, 44)
_ZONE_OCCUPIED_COLOR = (214, 39, 40)


def track_color(tid: int) -> tuple[int, int, int]:
    """Stable display color for a track id."""
    return _TRACK_COLORS[tid % len(_TRACK_COLORS)]


def height_colors(
    z: np.ndarray, zmin: float = _HEIGHT_RANGE_M[0], zmax: float = _HEIGHT_RANGE_M[1]
) -> np.ndarray:
    """Map point heights (m) to RGB colors, array of shape (N, 3), uint8.

    Heights at or below ``zmin`` map to the warm end of the gradient and
    heights at or above ``zmax`` to the cool end, with linear interpolation
    in between.
    """
    z = np.asarray(z, dtype=float)
    if zmax <= zmin:
        zmax = zmin + 1e-6
    fraction = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    stops = np.asarray(_HEIGHT_GRADIENT, dtype=float)
    position = fraction * (len(stops) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    blend = (position - low)[:, np.newaxis]
    return ((1.0 - blend) * stops[low] + blend * stops[high]).astype(np.uint8)


def run_viewer(
    config: AppConfig,
    frames: Iterator[SmallObstacleFrame],
    on_frame: Callable[[SmallObstacleFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        config: Application configuration; the occupancy zones and the
            tracker boundary box are read from ``config.chirp_config_path``
            and the plot ranges derive from them (falling back to
            ``config.gui``).
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

    zones = read_zones(config.chirp_config_path)
    boxes = read_boundary_boxes(config.chirp_config_path)

    # Height color range: frame the zone Z extent if zones are defined.
    if zones:
        zmin = min(zone[4] for zone in zones.values())
        zmax = max(zone[5] for zone in zones.values())
    else:
        zmin, zmax = _HEIGHT_RANGE_M

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD small obstacle detection")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD small obstacle detection")
    top = window.addPlot(row=0, col=0, title="Obstacles (top view)")
    top.setLabel("bottom", "X (m)")
    top.setLabel("left", "Y (m)")
    top.setAspectLocked(True)
    top.showGrid(x=True, y=True)
    top.addLegend()

    side = window.addPlot(row=1, col=0, title="Elevation (side view)")
    side.setLabel("bottom", "Y (m)")
    side.setLabel("left", "Z (m)")
    side.showGrid(x=True, y=True)
    window.ci.layout.setRowStretchFactor(0, 3)
    window.ci.layout.setRowStretchFactor(1, 1)

    # Plot ranges: frame the zones (or the boundary box) if configured.
    frame_box = None
    if zones:
        frame_box = (
            min(zone[0] for zone in zones.values()),
            max(zone[1] for zone in zones.values()),
            min(zone[2] for zone in zones.values()),
            max(zone[3] for zone in zones.values()),
        )
    elif "boundaryBox" in boxes:
        frame_box = boxes["boundaryBox"][:4]
    if frame_box is not None:
        xmin, xmax, ymin, ymax = frame_box
        top.setXRange(xmin - _ZONE_MARGIN_M, xmax + _ZONE_MARGIN_M, padding=0)
        top.setYRange(min(ymin, 0.0), ymax + _ZONE_MARGIN_M, padding=0)
        side.setXRange(min(ymin, 0.0), ymax + _ZONE_MARGIN_M, padding=0)
    else:
        top.setXRange(config.gui.x_range[0], config.gui.x_range[1], padding=0)
        top.setYRange(config.gui.y_range[0], config.gui.y_range[1], padding=0)
        side.setXRange(config.gui.y_range[0], config.gui.y_range[1], padding=0)
    side.setYRange(zmin - _ZONE_MARGIN_M, zmax + _ZONE_MARGIN_M, padding=0)

    # Qt6 moved the pen styles into the PenStyle enum; Qt5 has them flat.
    pen_styles = getattr(QtCore.Qt, "PenStyle", QtCore.Qt)
    if "boundaryBox" in boxes:
        xmin, xmax, ymin, ymax = boxes["boundaryBox"][:4]
        pen = pg.mkPen(color=(120, 120, 120), width=2, style=pen_styles.DashLine)
        top.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            pen=pen,
            name="Boundary box",
        )

    # One rectangle per occupancy zone in each view; the pen and fill are
    # switched per frame between the clear and occupied styles.
    zone_items: dict[int, tuple[object, object, object]] = {}
    for index, zone in sorted(zones.items()):
        xmin, xmax, ymin, ymax, z0, z1 = zone
        top_item = pg.PlotCurveItem(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            pen=pg.mkPen(color=_ZONE_CLEAR_COLOR, width=2),
            name=f"Zone {index}",
        )
        top.addItem(top_item)
        side_item = pg.PlotCurveItem(
            [ymin, ymax, ymax, ymin, ymin],
            [z0, z0, z1, z1, z0],
            pen=pg.mkPen(color=_ZONE_CLEAR_COLOR, width=2),
        )
        side.addItem(side_item)
        zone_label = pg.TextItem(f"zone {index}", color=_ZONE_CLEAR_COLOR)
        zone_label.setPos(xmin, ymax)
        top.addItem(zone_label)
        zone_items[index] = (top_item, side_item, zone_label)

    top_points = pg.ScatterPlotItem(pen=None, name="Points (color = height)")
    top.addItem(top_points)
    side_points = pg.ScatterPlotItem(pen=None)
    side.addItem(side_points)
    obstacles_scatter = pg.ScatterPlotItem(pen=pg.mkPen("k", width=1))
    top.addItem(obstacles_scatter)
    track_labels: dict[int, object] = {}

    def _set_zone_state(index: int, occupied: bool) -> None:
        color = _ZONE_OCCUPIED_COLOR if occupied else _ZONE_CLEAR_COLOR
        top_item, side_item, zone_label = zone_items[index]
        pen = pg.mkPen(color=color, width=5 if occupied else 2)
        top_item.setPen(pen)
        side_item.setPen(pen)
        zone_label.setColor(pg.mkColor(*color))
        zone_label.setText(f"zone {index}: {'OCCUPIED' if occupied else 'clear'}")

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return

        if len(frame.points):
            x, y, z = points_to_xyz(frame.points)
            colors = height_colors(z, zmin, zmax)
            brushes = [pg.mkBrush(*color, 180) for color in colors]
            sizes = point_sizes(frame.points[:, 4])
            top_points.setData(x=x, y=y, size=sizes, brush=brushes)
            side_points.setData(x=y, y=z, size=sizes, brush=brushes)
        else:
            top_points.setData(x=[], y=[])
            side_points.setData(x=[], y=[])

        spots = [
            {
                "pos": (obstacle.position[0], obstacle.position[1]),
                "size": 22,
                "brush": pg.mkBrush(*track_color(obstacle.tid), 180),
            }
            for obstacle in frame.obstacles
        ]
        obstacles_scatter.setData(spots)

        seen = set()
        for obstacle in frame.obstacles:
            seen.add(obstacle.tid)
            label = track_labels.get(obstacle.tid)
            if label is None:
                label = pg.TextItem(str(obstacle.tid), color="k", anchor=(0.5, 0.5))
                top.addItem(label)
                track_labels[obstacle.tid] = label
            label.setPos(obstacle.position[0], obstacle.position[1])
        for tid in list(track_labels):
            if tid not in seen:
                top.removeItem(track_labels.pop(tid))

        for index in zone_items:
            _set_zone_state(index, frame.zone_occupied(index))

        occupied = frame.occupied_zones()
        status = (
            f"ZONE {', '.join(str(zone) for zone in occupied)} OCCUPIED"
            if occupied
            else "all zones clear"
        )
        window.setWindowTitle(
            f"uRAD small obstacle detection — {status} | "
            f"{len(frame.obstacles)} obstacles, {len(frame.points)} points"
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

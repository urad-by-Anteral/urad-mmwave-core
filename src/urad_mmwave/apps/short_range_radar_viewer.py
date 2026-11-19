"""Live top view for the Short Range Radar application (optional GUI extra).

Draws the detection grid (1 m range rings and 30° sectors inside the field
of view), fills every detected zone with a color that encodes proximity —
red for the nearest third of the monitored range, orange for the middle
third and yellow for the farthest — and plots the point cloud on top.

Same threading model as the other viewers: frames are read and processed
in a background thread and the Qt timer draws only the latest one.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator

import numpy as np

from urad_mmwave.apps.short_range_radar import (
    GRID_HALF_SPAN_DEG,
    SECTOR_WIDTH_DEG,
    GridConfig,
)
from urad_mmwave.parser import Frame
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ARC_STEPS = 24
PROXIMITY_COLORS = ((220, 40, 40), (245, 150, 30), (240, 220, 40))
"""Zone fill colors: nearest, middle and farthest third of the range."""


def proximity_level(ring: int, max_range: float) -> int:
    """0 (red), 1 (orange) or 2 (yellow) for a ring's inner radius."""
    third = max_range / 3
    if ring < third:
        return 0
    if ring < 2 * third:
        return 1
    return 2


def sector_angles(sector: int) -> tuple[float, float]:
    """Azimuth limits (degrees from boresight) of a sector."""
    low = -GRID_HALF_SPAN_DEG + sector * SECTOR_WIDTH_DEG
    return low, low + SECTOR_WIDTH_DEG


def zone_outline(
    ring: int, sector: int, max_range: float
) -> tuple[np.ndarray, np.ndarray]:
    """X/Y vertices of a grid cell (annular sector), clipped to max_range."""
    r0, r1 = ring, min(ring + 1, max_range)
    a0, a1 = np.radians(sector_angles(sector))
    outer = np.linspace(a0, a1, _ARC_STEPS)
    inner = outer[::-1]
    angles = np.concatenate([outer, inner])
    radii = np.concatenate([np.full(_ARC_STEPS, r1), np.full(_ARC_STEPS, r0)])
    return radii * np.sin(angles), radii * np.cos(angles)


def run_viewer(
    frames: Iterator[Frame],
    grid: GridConfig,
    process: Callable[[Frame], np.ndarray],
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        frames: Iterator of decoded out-of-box frames (radar or replay).
        grid: Monitored area and alarm filter.
        process: Called for every frame from the reader thread; returns the
            ``(NUM_RINGS, NUM_SECTORS)`` detected-zone map (the CLI uses it
            for printing and file output too).
    """
    pg, QtCore = _import_pyqtgraph()
    from pyqtgraph.Qt import QtGui, QtWidgets

    latest: deque = deque(maxlen=1)
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in frames:
                latest.append((frame, process(frame)))
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOption("antialias", True)
    title = "uRAD Short Range Radar"
    app = pg.mkQApp(title)
    window = pg.GraphicsLayoutWidget(show=True, title=title)
    window.resize(900, 650)
    plot = window.addPlot(title="Top view")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    rmax = grid.max_range
    half_fov = np.radians(grid.fov / 2)
    xspan = rmax * max(np.sin(half_fov), 0.3)
    plot.setXRange(-xspan - 0.3, xspan + 0.3, padding=0)
    plot.setYRange(0, rmax + 0.3, padding=0)
    plot.setAspectLocked(True)

    # Zone polygons (hidden until detected), drawn below the grid lines
    no_pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
    zones: dict[tuple[int, int], QtWidgets.QGraphicsPolygonItem] = {}
    for ring in range(grid.num_rings):
        for sector in grid.sectors:
            xs, ys = zone_outline(ring, sector, rmax)
            polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in zip(xs, ys)])
            item = QtWidgets.QGraphicsPolygonItem(polygon)
            color = PROXIMITY_COLORS[proximity_level(ring, rmax)]
            item.setBrush(QtGui.QBrush(QtGui.QColor(*color, 170)))
            item.setPen(no_pen)
            item.setVisible(False)
            plot.addItem(item)
            zones[(ring, sector)] = item

    # Grid: arcs every meter and the sector boundaries inside the FoV
    grid_pen = pg.mkPen((40, 150, 70), width=1)
    first, last = grid.sectors[0], grid.sectors[-1]
    a_low, a_high = sector_angles(first)[0], sector_angles(last)[1]
    arc = np.radians(np.linspace(a_low, a_high, 120))
    radii = [float(r) for r in range(1, grid.num_rings)] + [rmax]
    for r in radii:
        plot.plot(r * np.sin(arc), r * np.cos(arc), pen=grid_pen)
    for sector in [*grid.sectors, last + 1]:
        a = np.radians(-GRID_HALF_SPAN_DEG + sector * SECTOR_WIDTH_DEG)
        plot.plot([0, rmax * np.sin(a)], [0, rmax * np.cos(a)], pen=grid_pen)

    scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(20, 20, 20, 220))
    plot.addItem(scatter)

    def _refresh() -> None:
        try:
            frame, detected = latest.pop()
        except IndexError:
            return
        for (ring, sector), item in zones.items():
            item.setVisible(bool(detected[ring, sector]))
        if len(frame.points):
            scatter.setData(
                x=frame.points[:, 0],
                y=frame.points[:, 1],
                size=point_sizes(frame.points[:, 4]),
            )
        else:
            scatter.setData(x=[], y=[])
        n_zones = int(np.count_nonzero(detected[: grid.num_rings]))
        window.setWindowTitle(
            f"{title} — frame {frame.header.frame_number} "
            f"({len(frame.points)} points, {n_zones} zone(s) detected)"
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
    # Let the reader leave the serial read before the caller closes the ports.
    reader.join(timeout=2.0)

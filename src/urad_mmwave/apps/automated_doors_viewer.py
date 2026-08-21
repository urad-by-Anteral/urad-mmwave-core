"""Live 2D top-view for the Automated Doors application (gui extra).

Shows the dynamic point cloud (marker size encodes SNR), the newly added
static points (orange squares), the tracked people/objects as numbered
circles colored per track id, the configured approach zone and obstruction
radius, and a prominent DOOR OPEN / DOOR CLOSED indicator at the sensor
position that turns green while a track triggers the door and yellow while
a static obstruction is detected.

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

from urad_mmwave.apps.automated_doors import (
    AutomatedDoorsFrame,
    DoorConfig,
    points_to_xy,
)
from urad_mmwave.apps.people_tracking_viewer import track_color
from urad_mmwave.config import AppConfig
from urad_mmwave.viewer import _import_pyqtgraph, point_sizes

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ZONE_MARGIN_M = 1.0

# Door indicator geometry (meters, drawn just below the sensor at y = 0).
_DOOR_WIDTH_M = 2.4
_DOOR_HEIGHT_M = 0.8

_DOOR_COLORS = {
    "open": (44, 160, 44),  # green
    "obstructed": (230, 180, 0),  # yellow
    "closed": (214, 39, 40),  # red
}


def door_color(door_open: bool, obstructed: bool) -> tuple[int, int, int]:
    """Indicator color mirroring the TI visualizer (green/yellow/red)."""
    if door_open:
        return _DOOR_COLORS["open"]
    if obstructed:
        return _DOOR_COLORS["obstructed"]
    return _DOOR_COLORS["closed"]


def run_viewer(
    config: AppConfig,
    frames: Iterator[AutomatedDoorsFrame],
    door: DoorConfig | None = None,
    on_frame: Callable[[AutomatedDoorsFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        config: Application configuration (``config.gui`` is unused here;
            the plot ranges derive from the approach zone).
        frames: Iterator of decoded frames with the door decision already
            applied (see :class:`~urad_mmwave.apps.automated_doors.DoorStateMachine`).
        door: Door trigger parameters used to draw the approach zone and
            obstruction radius (defaults match the firmware).
        on_frame: Optional callback invoked for every frame from the reader
            thread (used by the CLI for printing and file output).

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()
    door = door or DoorConfig()

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
    app = pg.mkQApp("uRAD automated doors")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD automated doors")
    plot = window.addPlot(title="Automated doors (top view)")
    plot.setLabel("bottom", "X (m)")
    plot.setLabel("left", "Y (m)")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    plot.addLegend()

    plot.setXRange(
        -door.half_width - _ZONE_MARGIN_M, door.half_width + _ZONE_MARGIN_M, padding=0
    )
    plot.setYRange(-_DOOR_HEIGHT_M - 0.4, door.depth + _ZONE_MARGIN_M, padding=0)

    # Qt6 moved the pen styles into the PenStyle enum; Qt5 has them flat.
    pen_styles = getattr(QtCore.Qt, "PenStyle", QtCore.Qt)

    # Approach zone: tracks inside it moving towards the door trigger it.
    zone_pen = pg.mkPen(color=(31, 119, 180), width=2, style=pen_styles.DashLine)
    plot.plot(
        [-door.half_width, door.half_width, door.half_width, -door.half_width,
         -door.half_width],
        [0.0, 0.0, door.depth, door.depth, 0.0],
        pen=zone_pen,
        name="Approach zone",
    )

    # Obstruction radius: static objects inside it flag an obstruction.
    theta = np.linspace(0.0, np.pi, 60)
    obstruction_pen = pg.mkPen(color=(255, 130, 48), width=2, style=pen_styles.DotLine)
    plot.plot(
        door.obstruction_range * np.cos(theta),
        door.obstruction_range * np.sin(theta),
        pen=obstruction_pen,
        name="Obstruction range",
    )

    dynamic_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(20, 20, 20, 140), name="Dynamic points"
    )
    plot.addItem(dynamic_scatter)
    static_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(255, 130, 48, 200), symbol="s", size=8,
        name="Static points",
    )
    plot.addItem(static_scatter)
    targets_scatter = pg.ScatterPlotItem(pen=pg.mkPen("k", width=1))
    plot.addItem(targets_scatter)
    track_labels: dict[int, object] = {}

    # Door indicator: a filled bar at the sensor position plus a state label.
    door_bar = pg.BarGraphItem(
        x=[0.0],
        y0=[-_DOOR_HEIGHT_M],
        height=[_DOOR_HEIGHT_M],
        width=[_DOOR_WIDTH_M],
        brush=pg.mkBrush(*_DOOR_COLORS["closed"]),
        pen=pg.mkPen("k", width=1),
    )
    plot.addItem(door_bar)
    door_text = pg.TextItem(anchor=(0.5, 0.5), color="w")
    door_text.setPos(0.0, -_DOOR_HEIGHT_M / 2)
    plot.addItem(door_text)

    def _set_door_text(label: str) -> None:
        door_text.setHtml(
            f"<div style='text-align:center; font-size:14pt; font-weight:bold; "
            f"color:white'>DOOR {label}</div>"
        )

    _set_door_text("CLOSED")

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

        door_bar.setOpts(brush=pg.mkBrush(*door_color(frame.door_open, frame.obstructed)))
        _set_door_text("OPEN" if frame.door_open else "CLOSED")

        obstruction = ", STATIC OBSTRUCTION" if frame.obstructed else ""
        window.setWindowTitle(
            f"uRAD automated doors — door {frame.door_label}, "
            f"{len(frame.targets)} tracked, "
            f"{len(frame.dynamic_points)} dynamic / "
            f"{len(frame.static_points)} static points{obstruction}"
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

"""Live cabin view for the CPD with Classification application (gui extra).

Top view of the car cabin (X = width, driver side positive; Y = depth,
rear positive): the configured occupancy zones are drawn as rectangles
whose fill color encodes the state (empty / occupied / adult / child), with
a label per zone showing the state, the point count and — once a
classification decision is available — the decision metrics (accumulated
SNR in dB and volume variance). The transformed point cloud is overlaid;
the marker size encodes SNR.

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

from urad_mmwave.apps.cpd import CabinSetup, CpdFrame, ZoneStatus
from urad_mmwave.viewer import _import_pyqtgraph

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 33
_ZONE_MARGIN_M = 0.5
_MIN_POINT_SIZE = 4.0
_MAX_POINT_SIZE = 16.0

# Zone state -> RGB color (rectangle border; the fill uses _FILL_ALPHA).
ZONE_STATE_COLORS = {
    "empty": (150, 150, 150),
    "occupied": (255, 170, 0),
    "adult": (31, 119, 180),
    "child": (214, 39, 40),
}
_FILL_ALPHA = {"empty": 20, "occupied": 90, "adult": 110, "child": 110}


def zone_state_color(label: str) -> tuple[int, int, int]:
    """Display color for a zone state label (unknown states render gray)."""
    return ZONE_STATE_COLORS.get(label, ZONE_STATE_COLORS["empty"])


def zone_label_text(status: ZoneStatus) -> str:
    """Zone annotation: state, point count and classification metrics."""
    text = f"Z{status.zone_id} {status.label.upper()}"
    if status.occupied:
        text += f"\n{status.num_points} pts"
        if status.decision_snr_db is not None:
            text += f"\nsnr {status.decision_snr_db:.1f} dB"
        if status.decision_volume is not None:
            text += f", vol {status.decision_volume:.3f}"
    return text


def point_sizes(snr: np.ndarray) -> np.ndarray:
    """Map linear firmware SNR values to marker sizes in pixels."""
    return np.clip(_MIN_POINT_SIZE + snr / 8.0, _MIN_POINT_SIZE, _MAX_POINT_SIZE)


def run_viewer(
    setup: CabinSetup,
    frames: Iterator[CpdFrame],
    on_frame: Callable[[CpdFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        setup: Cabin setup (zone cuboids and interior bounds) parsed from
            the chirp configuration; NULL zones are not drawn.
        frames: Iterator of decoded frames with the zone states already
            computed (i.e. after :meth:`OccupancyTracker.update`).
        on_frame: Optional callback invoked for every frame from the reader
            thread (used by the CLI for printing and file output).

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()
    from pyqtgraph.Qt import QtWidgets

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
    app = pg.mkQApp("uRAD CPD")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD CPD")
    plot = window.addPlot(title="Cabin occupancy (top view)")
    plot.setLabel("bottom", "X (m) — driver side positive")
    plot.setLabel("left", "Y (m) — rear positive")
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)
    plot.invertX(True)  # match the TI visualizer: driver side drawn left

    drawn_zones = [zone for zone in setup.zones if not zone.is_null]
    if setup.interior_bounds is not None:
        xmin, xmax, ymin, ymax = setup.interior_bounds
    else:
        xmin = min(zone.bounds_xy()[0] for zone in drawn_zones)
        xmax = max(zone.bounds_xy()[1] for zone in drawn_zones)
        ymin = min(zone.bounds_xy()[2] for zone in drawn_zones)
        ymax = max(zone.bounds_xy()[3] for zone in drawn_zones)
    plot.setXRange(xmin - _ZONE_MARGIN_M, xmax + _ZONE_MARGIN_M, padding=0)
    plot.setYRange(ymin - _ZONE_MARGIN_M, ymax + _ZONE_MARGIN_M, padding=0)

    # One rectangle + one text label per (non-NULL) zone, restyled per frame.
    zone_rects: dict[int, object] = {}
    zone_labels: dict[int, object] = {}
    for zone in drawn_zones:
        zxmin, zxmax, zymin, zymax = zone.bounds_xy()
        rect = QtWidgets.QGraphicsRectItem(zxmin, zymin, zxmax - zxmin, zymax - zymin)
        rect.setPen(pg.mkPen(color=zone_state_color("empty"), width=2))
        rect.setBrush(pg.mkBrush(*zone_state_color("empty"), _FILL_ALPHA["empty"]))
        plot.addItem(rect)
        zone_rects[zone.zone_id] = rect

        label = pg.TextItem(f"Z{zone.zone_id}", color="k", anchor=(0.5, 0.5))
        label.setPos((zxmin + zxmax) / 2, (zymin + zymax) / 2)
        plot.addItem(label)
        zone_labels[zone.zone_id] = label

    points_scatter = pg.ScatterPlotItem(
        pen=None, brush=pg.mkBrush(20, 20, 20, 140), name="Points"
    )
    plot.addItem(points_scatter)

    def _refresh() -> None:
        try:
            frame = latest.pop()
        except IndexError:
            return

        if len(frame.cabin_points):
            points_scatter.setData(
                x=frame.cabin_points[:, 0],
                y=frame.cabin_points[:, 1],
                size=point_sizes(frame.cabin_points[:, 3]),
            )
        else:
            points_scatter.setData(x=[], y=[])

        for status in frame.zones:
            rect = zone_rects.get(status.zone_id)
            label = zone_labels.get(status.zone_id)
            if rect is None or label is None:
                continue  # NULL zone, not drawn
            color = zone_state_color(status.label)
            alpha = _FILL_ALPHA.get(status.label, _FILL_ALPHA["empty"])
            rect.setPen(pg.mkPen(color=color, width=2))
            rect.setBrush(pg.mkBrush(*color, alpha))
            label.setText(zone_label_text(status))

        occupied = sum(status.occupied for status in frame.zones)
        window.setWindowTitle(
            f"uRAD CPD — {occupied}/{len(frame.zones)} zones occupied, "
            f"{len(frame.points)} points, frame {frame.frame_number}"
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

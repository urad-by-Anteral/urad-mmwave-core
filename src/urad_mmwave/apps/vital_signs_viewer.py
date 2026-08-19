"""Live waveform view for the vital signs application (gui extra).

Shows the heart and breathing waveforms streamed by the firmware (each
vital signs TLV carries the latest 15 samples of both) as two scrolling
plots, with the patient status, the median-smoothed heart rate and the
breathing rate in the window title.

Frames are read from the radar in a background thread; the Qt main
thread draws only the samples received so far, so a slow display never
backs up the serial stream.

Requires the ``gui`` extra: ``pip install urad-mmwave[gui]``.
"""

from __future__ import annotations

import logging
import statistics
import threading
from collections import deque
from collections.abc import Callable, Iterator

from urad_mmwave.apps.vital_signs import (
    HOLDING_BREATH_THRESHOLD,
    VitalSignsFrame,
    patient_status,
)
from urad_mmwave.viewer import _import_pyqtgraph

log = logging.getLogger(__name__)

_REFRESH_INTERVAL_MS = 100
_WAVEFORM_WINDOW_SAMPLES = 450  # 30 vitals updates of 15 samples each

_HEART_COLOR = (230, 25, 75)
_BREATH_COLOR = (67, 99, 216)


def run_viewer(
    frames: Iterator[VitalSignsFrame],
    on_frame: Callable[[VitalSignsFrame], None] | None = None,
) -> None:
    """Open the viewer window and stream frames until it is closed.

    Args:
        frames: Iterator of decoded vital signs frames.
        on_frame: Optional callback invoked for every frame from the reader
            thread (used by the CLI for printing and file output).

    Raises:
        ImportError: If pyqtgraph or a Qt binding is not installed.
    """
    pg, QtCore = _import_pyqtgraph()

    latest: deque = deque(maxlen=1)
    heart_wave: deque = deque(maxlen=_WAVEFORM_WINDOW_SAMPLES)
    breath_wave: deque = deque(maxlen=_WAVEFORM_WINDOW_SAMPLES)
    recent_heart_rates: deque = deque(maxlen=10)
    lock = threading.Lock()
    stop_reading = threading.Event()

    def _read_loop() -> None:
        try:
            for frame in frames:
                if on_frame is not None:
                    on_frame(frame)
                if frame.vitals is not None:
                    with lock:
                        heart_wave.extend(frame.vitals.heart_waveform)
                        breath_wave.extend(frame.vitals.breath_waveform)
                        if frame.vitals.breathing_deviation >= HOLDING_BREATH_THRESHOLD:
                            recent_heart_rates.append(frame.vitals.heart_rate)
                latest.append(frame)
                if stop_reading.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surface errors from the thread
            if not stop_reading.is_set():
                log.error("Frame reader stopped: %s", exc)

    reader = threading.Thread(target=_read_loop, name="urad-frame-reader", daemon=True)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp("uRAD vital signs")
    window = pg.GraphicsLayoutWidget(show=True, title="uRAD vital signs")

    heart_plot = window.addPlot(title="Heart waveform")
    heart_plot.setLabel("left", "Amplitude")
    heart_plot.showGrid(x=True, y=True)
    heart_curve = heart_plot.plot(pen=pg.mkPen(color=_HEART_COLOR, width=2))

    window.nextRow()
    breath_plot = window.addPlot(title="Breathing waveform")
    breath_plot.setLabel("left", "Amplitude")
    breath_plot.setLabel("bottom", "Sample")
    breath_plot.showGrid(x=True, y=True)
    breath_curve = breath_plot.plot(pen=pg.mkPen(color=_BREATH_COLOR, width=2))

    def _refresh() -> None:
        with lock:
            heart = list(heart_wave)
            breath = list(breath_wave)
            rates = list(recent_heart_rates)
        try:
            frame = latest.pop()
        except IndexError:
            frame = None

        heart_curve.setData(heart)
        breath_curve.setData(breath)

        if frame is not None:
            title = f"uRAD vital signs — patient: {patient_status(frame)}"
            if frame.vitals is not None and rates:
                title += (
                    f"  |  heart: {statistics.median(rates):.1f} bpm"
                    f"  |  breath: {frame.vitals.breathing_rate:.1f} rpm"
                )
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

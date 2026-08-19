# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Live waveform view for vital signs: `urad-vital-signs --gui` plots the
  heart and breathing waveforms streamed by the firmware as two scrolling
  charts, with the patient status, median-smoothed heart rate and
  breathing rate in the window title.
- Live range profile for level sensing: `urad-level-sensing --spectrum`
  enables the raw ADC stream (TLV type 2, second `guiMonitor` flag) and
  shows the full FFT range profile with the three detected peaks marked
  and their amplitudes — the modern equivalent of the legacy
  `plotSpectrum` GUI, with the same processing (DC removal, Hanning,
  4096-point FFT). New APIs: `decode_iq()`, `range_spectrum()`, and
  `sampling_rate`/`raw_iq` settings (frame rate was fixed at 20 Hz).
- Live distance view for level sensing: `urad-level-sensing --gui` streams
  continuously (new `stream()` API alongside the one-shot `measure()`) and
  plots the three high-accuracy ranges over time with the latest values in
  the window title.
- Live top view for people tracking: `urad-people-tracking --gui` shows
  the point cloud (SNR-sized markers), the tracked people as numbered
  circles colored per track id, and the tracker zones defined in the
  chirp configuration (`boundaryBox`, `staticBoundaryBox`,
  `presenceBoundaryBox`); the plot is framed to the boundary box.

### Fixed
- Level sensing range 3 no longer reads 62.5 mm short when the low word of
  its fixed-point value is >= 0x8000: the legacy decode treated that word
  as signed (a leftover of the out-of-box point struct, where the slot is
  a signed doppler index). Ranges 1 and 2 were always decoded correctly.

## [0.2.0] - 2026-08-19

All changes in this release were validated against a uRAD Industrial
(IWR6843AoP) over USB.

### Added
- `urad-vital-signs` CLI and `urad_mmwave.apps.vital_signs` module for the
  TI "Vital Signs with People Tracking" firmware (Radar Toolbox): decodes
  the vital signs TLV (type 1040) — heart rate, breathing rate, breathing
  deviation and both waveform buffers — alongside the shared tracking TLVs,
  prints a patient status (measuring / present / holding breath) with
  median-smoothed heart rate, and writes VitalSigns/PointCloud/Targets
  output files.

### Changed
- Renamed the people counting application to people tracking, following the
  TI Radar Toolbox naming: the CLI is now `urad-people-tracking` (was
  `urad-people-counting`), the module `urad_mmwave.apps.people_tracking`
  (was `.people_counting`) and the frame class `PeopleTrackingFrame` (was
  `PeopleCountingFrame`). The UART protocol and output formats are
  unchanged.

### Fixed
- The people tracking and vital signs parsers now recognize the 0xBE
  end-of-frame alignment padding emitted by the Radar Toolbox firmwares,
  instead of logging an "implausible TLV" warning on every frame.
- The live viewer no longer logs a spurious serial error on Windows when its
  window is closed: the frame reader thread is now joined before the serial
  ports are closed underneath it.
- `RadarSession` now fails fast with a clear error when the radar does not
  respond to any configuration command (typically after a previous session,
  since application firmwares accept only one configuration per boot and
  need a power cycle), instead of silently continuing towards a hang.
- `RadarSession.packets()` now raises `StreamTimeoutError` after ~30 s
  without data by default (`max_empty_reads=100`) instead of waiting
  forever, so application CLIs exit with a useful error when the sensor
  never starts.

## [0.1.0] - 2026-08-19

First public release, unifying the Python code previously duplicated across
the uRAD Automotive, Automotive HPA and Industrial SDKs.

### Added
- `urad_mmwave` package: typed JSON configuration (`load_config`), radar
  session lifecycle (`RadarSession`), TLV stream parser (`read_frames`) and
  output writers (`PointCloudWriter`, `TemperatureWriter`).
- `urad-mmwave` CLI with port overrides, single-UART mode, duration/frame
  limits and verbose logging.
- Support for TLV types 1 (detected points), 7 (side info) and 9
  (temperature); unknown TLVs are skipped safely.
- Raspberry Pi single-UART support with optional GPIO chip reset
  (`pip install urad-mmwave[rpi]`).
- Hardware-independent test suite with synthetic TLV packets; CI on
  Linux/Windows for Python 3.9 and 3.13.
- Live 2D point cloud viewer (`--gui`, `pip install urad-mmwave[gui]`) with
  SNR-scaled markers and per-product plot ranges, replacing the legacy
  `out_of_box_demo_USB_GUI.py` scripts.
- Product profiles under `profiles/` (automotive, automotive-hpa,
  industrial): serial settings, GUI ranges and the chirp configurations
  (standard + temperature) for each product.
- Relative `chirp_config_path` values are resolved against the JSON config
  file's directory, so profiles work from any working directory.
- Application clients: `urad-level-sensing` (high accuracy level sensing for
  AWR/IWR with generated chirp configuration, fixed-point range decoding and
  averaged measurements) and `urad-people-counting` (3D people counting,
  standard and overhead, decoding target list/index/height, compressed point
  cloud and presence TLVs).
- Generic `iter_packets()` primitive and `RadarSession.packets()` for
  application firmwares that share the packet framing but define their own
  TLV sets; bounded idle timeout via `max_empty_reads`.
- People counting doppler values are now decoded as signed (the legacy
  scripts decoded them as unsigned, wrapping negative velocities).

### Fixed (relative to the legacy per-product scripts)
- Unknown TLV types no longer desynchronize the parser.
- Point cloud and side-info TLVs are paired correctly regardless of TLV
  order within the frame.
- Serial reads are bounded: a stalled stream raises a timeout instead of
  hanging forever.
- `sensorStop` is always sent and ports are always closed on exit
  (including Ctrl+C), so the radar does not keep transmitting.
- All values in the JSON configuration (baudrates, timeouts, paths) are
  actually honoured at runtime.
- Output files are kept open for the session instead of being reopened on
  every frame.

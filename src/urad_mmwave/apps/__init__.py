"""Application-specific modules for uRAD mmWave products.

Each application targets its own firmware (distributed as release assets in
the product repositories) and builds on the generic packet framing of
:mod:`urad_mmwave.parser`:

- :mod:`urad_mmwave.apps.level_sensing` — High Accuracy Level Sensing
  (uRAD Automotive and uRAD Industrial).
- :mod:`urad_mmwave.apps.people_tracking` — 3D People Tracking (formerly
  People Counting), standard and overhead (uRAD Industrial).
- :mod:`urad_mmwave.apps.vital_signs` — Vital Signs with People Tracking
  (uRAD Industrial).
"""

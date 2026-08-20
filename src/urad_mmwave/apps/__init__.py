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
- :mod:`urad_mmwave.apps.area_scanner` — Area Scanner with static object
  detection and safety zones (uRAD Industrial).
- :mod:`urad_mmwave.apps.automated_doors` — Automated Doors and Gates
  trigger logic (uRAD Industrial).
- :mod:`urad_mmwave.apps.small_obstacle` — Small Obstacle Detection for
  mobile robots (uRAD Industrial).
- :mod:`urad_mmwave.apps.cpd` — in-cabin occupancy / Child Presence
  Detection with adult-child classification (uRAD Industrial).
- :mod:`urad_mmwave.apps.medium_range_radar` — Medium Range Radar ADAS
  demo with clustering, tracking and parking assist (uRAD Automotive).
"""

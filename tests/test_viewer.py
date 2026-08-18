"""Unit tests for the viewer helpers that do not require Qt."""

from __future__ import annotations

import numpy as np

from urad_mmwave.viewer import point_sizes


def test_point_sizes_clipped_to_bounds():
    snr = np.array([0, 200, 100000])
    sizes = point_sizes(snr)
    assert sizes[0] == 4.0  # minimum size
    assert sizes[-1] == 16.0  # maximum size
    assert np.all(sizes >= 4.0)
    assert np.all(sizes <= 16.0)


def test_point_sizes_monotonic():
    snr = np.array([50, 100, 200, 400])
    sizes = point_sizes(snr)
    assert np.all(np.diff(sizes) >= 0)

#!/usr/bin/env python3
"""Configuration for the authoritative Figure 4 workflow."""

from __future__ import annotations

import os


LATTICE_SIZE = 15
ALPHA_VALUES = [round(value / 10, 1) for value in range(1, 21)]
ENSEMBLE_SIZE = 100
VORONOI_ENSEMBLE_SIZE = 1000
EPSILON = 1e-3
BASE_SEED = 20260713
DEFAULT_WORKERS = min(48, os.cpu_count() or 1)

DATA_OUTPUT_FILE = "mixing_times_vs_alpha_data.json"
LOG_PLOT_FILE = "mixing_times_vs_alpha_log.png"

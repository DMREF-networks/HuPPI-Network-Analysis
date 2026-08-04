#!/usr/bin/env python3

"""
config.py

Central configuration file for network data generation and plotting.
This file contains all parameters needed for both generate_data.py and plot.py.
"""

###############################################################################
#                    SHARED PARAMETERS
###############################################################################
# Directory setup
BASE_DIR = "data"         # Base directory for raw data and analysis outputs
OUTPUT_DIR = "plots"      # Directory for saving plots

# Core parameters
LATTICE_SIZE = 15         # Lattice size (e.g., 15 means a 15x15 grid)
DIMENSIONS = 2            # Spatial dimensions
ENSEMBLE_SIZE = 100        # Number of realizations per configuration

# Distribution and graph type configurations
DISTRIBUTION_CLASSES = ["ordered", "classI", "classII", "classIII", "random"]
GRAPH_TYPES = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi_pruned"]

# Map distribution types to their alpha values
DIST_TO_ALPHA_MAP = {
    "ordered": 0.0,
    "classI": 1.0,
    "classII": 1.0,
    "classIII": 1.0,
    "random": 999.0  # Filename sentinel; alpha does not affect Poisson patterns
}

###############################################################################
#                  GENERATE_DATA.PY SPECIFIC PARAMETERS
###############################################################################
# Parameters specific to data generation
ALPHA_FOR_ORDERED = 0.1
ALPHA_FOR_HYPERUNIFORM = 1.0
ORC_ALPHA = 0  # Parameter for Ollivier-Ricci calculations (if needed)
# (Optional) Lattice spacing parameter
C = 1.0

###############################################################################
#                       PLOT.PY SPECIFIC PARAMETERS
###############################################################################
# Plotting parameters
HISTOGRAM_BIN_COUNT = 30
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
LINE_WIDTH = 2.5
FONT_SIZE_LABEL = 18
TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 12
MARKER_SIZE = 7

# Figure sizes
OVERLAPPED_FIG_SIZE = (12, 10)
SUBPLOT_FIG_SIZE = (20, 15)

# Y-axis limits for different graph types (if you want to enforce specific limits)
Y_LIMITS = {
    "upper_graphs": (0, 4),  # For gabriel and delaunay
    "lower_graphs": (0, 2)   # For delaunay_centroidal and voronoi_pruned
}

###############################################################################
#                    HELPER FUNCTIONS
###############################################################################
def get_data_params():
    """Return parameters needed for generate_data.py."""
    return {
        "base_dir": BASE_DIR,
        "ensemble_size": ENSEMBLE_SIZE,
        "lattice_size": LATTICE_SIZE,
        "dimensions": DIMENSIONS,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "dist_to_alpha_map": DIST_TO_ALPHA_MAP,
        "alpha_for_ordered": ALPHA_FOR_ORDERED,
        "alpha_for_hyperuniform": ALPHA_FOR_HYPERUNIFORM,
        "orc_alpha": ORC_ALPHA,
        "C": C
    }

def get_plot_params():
    """Return parameters needed for plot.py."""
    return {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "ensemble_size": ENSEMBLE_SIZE,
        "lattice_size": LATTICE_SIZE,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "dist_to_alpha_map": DIST_TO_ALPHA_MAP,
        "bin_count": HISTOGRAM_BIN_COUNT,
        "dpi": FIGURE_DPI,
        "figure_size": FIGURE_SIZE,
        "line_width": LINE_WIDTH,
        "font_size_label": FONT_SIZE_LABEL,
        "tick_label_size": TICK_LABEL_SIZE,
        "legend_font_size": LEGEND_FONT_SIZE,
        "marker_size": MARKER_SIZE,
        "overlapped_fig_size": OVERLAPPED_FIG_SIZE,
        "subplot_fig_size": SUBPLOT_FIG_SIZE,
        "y_limits": Y_LIMITS
    }

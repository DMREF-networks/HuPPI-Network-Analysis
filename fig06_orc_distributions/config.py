#!/usr/bin/env python3

"""
config.py

Central configuration file for manuscript Figure 6 data generation and plotting.
This file contains all parameters needed for both generate_data.py and plot.py.
"""

###############################################################################
#                    SHARED PARAMETERS
###############################################################################
# Directory setup
BASE_DIR = "data"
OUTPUT_DIR = "plots"  # For plot outputs

# Core parameters
LATTICE_SIZE = 15
DIMENSIONS = 2
ENSEMBLE_SIZE = 50

# Distribution and graph type configurations
DISTRIBUTION_CLASSES = ["classI", "random"]
GRAPH_TYPES = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi_pruned"]

# Map distribution types to their alpha values
DIST_TO_ALPHA_MAP = {
    "ordered": 0.0,
    "classI": 1.0,
    "classII": 1.0,
    "classIII": 1.0,
    "random": 999.0
}

###############################################################################
#                  GENERATE_DATA.PY SPECIFIC PARAMETERS
###############################################################################
# Parameters specific to data generation
ALPHA_FOR_ORDERED = 0.1
ALPHA_FOR_HYPERUNIFORM = 1.0
ORC_ALPHA = 0  # Ollivier-Ricci parameter

###############################################################################
#                    PLOT.PY SPECIFIC PARAMETERS
###############################################################################
# Plotting parameters
HISTOGRAM_BIN_COUNT = 30
FIGURE_DPI = 300

# Figure sizes
OVERLAPPED_FIG_SIZE = (12, 10)
SUBPLOT_FIG_SIZE = (20, 15)

# Y-axis limits for different graph types
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
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "dist_to_alpha_map": DIST_TO_ALPHA_MAP,
        "ensemble_size": ENSEMBLE_SIZE,
        "lattice_size": LATTICE_SIZE,
        "dimensions": DIMENSIONS,
        "alpha_for_ordered": ALPHA_FOR_ORDERED,
        "alpha_for_hyperuniform": ALPHA_FOR_HYPERUNIFORM,
        "orc_alpha": ORC_ALPHA
    }

def get_plot_params():
    """Returns parameters needed for plot.py"""
    return {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "dist_to_alpha_map": DIST_TO_ALPHA_MAP,
        "ensemble_size": ENSEMBLE_SIZE,
        "lattice_size": LATTICE_SIZE,
        "bin_count": HISTOGRAM_BIN_COUNT,
        "dpi": FIGURE_DPI,
        "overlapped_fig_size": OVERLAPPED_FIG_SIZE,
        "subplot_fig_size": SUBPLOT_FIG_SIZE,
        "y_limits": Y_LIMITS
    }

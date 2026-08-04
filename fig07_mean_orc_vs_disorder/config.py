#!/usr/bin/env python3

"""
config.py

Central configuration file for Figure 7 data generation and plotting.
Contains parameters for alpha sweep experiments and visualization.
"""

import numpy as np

###############################################################################
#                    SHARED PARAMETERS
###############################################################################
# Directory setup
BASE_DIR = "data"
OUTPUT_DIR = "plots"

# Core parameters
DIMENSIONS = 2
ENSEMBLE_SIZE = 50
ORC_ALPHA = 0  # Ollivier-Ricci parameter

# Distribution and graph configurations
DISTRIBUTION_CLASSES = ["classI"]#, "classII", "classIII"]
GRAPH_TYPES = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi_pruned"]

# Alpha sweep parameters
ALPHA_VALUES = np.linspace(0.1, 2.0, 20)

###############################################################################
#                    GENERATE_DATA.PY SPECIFIC PARAMETERS
###############################################################################
# Lattice parameters
LATTICE_SIZE = 15

###############################################################################
#                    PLOT.PY SPECIFIC PARAMETERS
###############################################################################
# Plotting parameters
COLORS = {
    # Colors for graph types
    "delaunay": "#0077BB",
    "gabriel": "#EE7733",
    "delaunay_centroidal": "#009988",
    "voronoi_pruned": "#CC3311",
    # Colors for distribution classes
    "classI": "darkblue",
    "classII": "darkred",
    "classIII": "darkgreen",
    "random": "gray"
}

LINESTYLES = {
    "delaunay": "solid",
    "gabriel": "dashed",
    "delaunay_centroidal": "dashdot",
    "voronoi_pruned": "dotted"
}

# Figure sizes
STYLE1_FIG_SIZE = (12, 8)
STYLE2_FIG_SIZE = (15, 12)
STYLE3_FIG_SIZE = (15, 12)

# Plot parameters
FIGURE_DPI = 300
GRID_ALPHA = 0.3
CAPSIZE = 3
HIST_BINS = 15

###############################################################################
#                    HELPER FUNCTIONS
###############################################################################
def get_data_generation_params():
    """Returns parameters needed for generate_data.py"""
    return {
        "base_dir": BASE_DIR,
        "dimensions": DIMENSIONS,
        "ensemble_size": ENSEMBLE_SIZE,
        "orc_alpha": ORC_ALPHA,
        "alpha_values": ALPHA_VALUES,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "lattice_size": LATTICE_SIZE
    }

def get_plot_params():
    """Returns parameters needed for plot.py"""
    return {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "alpha_values": ALPHA_VALUES,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "ensemble_size": ENSEMBLE_SIZE,
        "colors": COLORS,
        "linestyles": LINESTYLES,
        "style1_fig_size": STYLE1_FIG_SIZE,
        "style2_fig_size": STYLE2_FIG_SIZE,
        "style3_fig_size": STYLE3_FIG_SIZE,
        "dpi": FIGURE_DPI,
        "grid_alpha": GRID_ALPHA,
        "capsize": CAPSIZE,
        "hist_bins": HIST_BINS
    }

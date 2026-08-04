#!/usr/bin/env python3

"""
config.py

Central configuration file for manuscript Figure 8 data generation and plotting.
Contains parameters for lattice size experiments and visualization.
"""

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

# Lattice size parameters (Assumed same for all graph types)
LATTICE_SIZE_RANGES = {
    "delaunay": range(5, 18),
    "gabriel": range(5, 18),
    "delaunay_centroidal": range(5, 15),
    "voronoi_pruned": range(5, 15)
}

# Distribution and graph configurations
# Removed "random" from distribution classes to avoid duplicate random pattern generation
DISTRIBUTION_CLASSES = ["classI"]
GRAPH_TYPES = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi_pruned"]

###############################################################################
#                    PLOT.PY SPECIFIC PARAMETERS
###############################################################################
# Plotting parameters
COLORS = {
    "delaunay": "#0077BB",
    "gabriel": "#EE7733",
    "delaunay_centroidal": "#009988",
    "voronoi_pruned": "#CC3311"
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

# Plot DPI
FIGURE_DPI = 300

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
        "lattice_size_ranges": LATTICE_SIZE_RANGES,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES
    }

def get_plot_params():
    """Returns parameters needed for plot.py"""
    return {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "ensemble_size": ENSEMBLE_SIZE,
        "lattice_size_ranges": LATTICE_SIZE_RANGES,
        "distribution_classes": DISTRIBUTION_CLASSES,
        "graph_types": GRAPH_TYPES,
        "colors": COLORS,
        "linestyles": LINESTYLES,
        "style1_fig_size": STYLE1_FIG_SIZE,
        "style2_fig_size": STYLE2_FIG_SIZE,
        "style3_fig_size": STYLE3_FIG_SIZE,
        "dpi": FIGURE_DPI
    }

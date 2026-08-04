#!/usr/bin/env python3
"""
config.py

Central configuration file for data generation and plotting.
Defines all parameters needed for generate_data.py and plot.py.
"""

# Directories
BASE_DIR = "data"         # Base directory for raw data and analysis outputs
OUTPUT_DIR = "plots"      # Directory for saving plots

# Core parameters for data generation
DIMENSIONS = 2            # Spatial dimensions
ENSEMBLE_SIZE = 100        # Number of realizations per configuration

# Data generation parameters
ALPHA = 1.0               # Perturbation strength stated in the Fig. 3 caption
C = 1.0                   # Lattice spacing parameter
BASE_SEED = 20260713      # Reproducible point-pattern seed stream

# Graph types to generate and analyze
GRAPH_TYPES = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi_pruned"]

# Define lattice size ranges for each graph type.
# For each graph type, the simulation will run for every lattice size in the list.
SIZE_RANGES = {
    "delaunay": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "gabriel": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "delaunay_centroidal": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "voronoi_pruned": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
}

# Colors for plotting (mapping each graph type to a color)
COLORS = {
    "delaunay": "#0077BB",
    "gabriel": "#EE7733",
    "delaunay_centroidal": "#009988",
    "voronoi_pruned": "#CC3311"
}

# Plotting parameters
FIGURE_SIZE = (12, 8)
DPI = 300
LINE_WIDTH = 2.5
FONT_SIZE_LABEL = 18
TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 16
MARKER_SIZE = 7

def get_data_params():
    """Returns parameters for data generation (generate_data.py)."""
    return {
        "base_dir": BASE_DIR,
        "ensemble_size": ENSEMBLE_SIZE,
        "alpha": ALPHA,
        "C": C,
        "base_seed": BASE_SEED,
        "graph_types": GRAPH_TYPES,
        "dimensions": DIMENSIONS,
        "size_ranges": SIZE_RANGES
    }

def get_plot_params():
    """Returns parameters for plotting (plot.py)."""
    return {
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "graph_types": GRAPH_TYPES,
        "colors": COLORS,
        "figure_size": FIGURE_SIZE,
        "dpi": DPI,
        "line_width": LINE_WIDTH,
        "font_size_label": FONT_SIZE_LABEL,
        "tick_label_size": TICK_LABEL_SIZE,
        "legend_font_size": LEGEND_FONT_SIZE,
        "marker_size": MARKER_SIZE
    }

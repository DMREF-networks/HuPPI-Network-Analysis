#!/usr/bin/env python3
"""Plot the network metrics computed for manuscript Figure 2.

This script loads the network metrics computed by generate_data.py and produces a plot of
Normalized Total Effective Resistance (TER) vs. perturbation strength (alpha) for different
graph types. For each graph type, the script plots the average TER for hyperuniform (classI)
patterns as a function of alpha and overlays a horizontal dashed line representing the
TER for random (Poisson) patterns.

Configuration parameters (e.g., directories, graph types, figure properties) are read from config.py.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config  # configuration file

# Get plotting parameters from config
plot_cfg = config.get_plot_params()
BASE_DIR = plot_cfg["base_dir"]
OUTPUT_DIR = plot_cfg["output_dir"]
GRAPH_TYPES = plot_cfg["graph_types"]

# RSA data path
RSA_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "rsa", "rsa_network_properties.json")

# Define a simple color mapping for the graph types
COLORS = {
    "delaunay": "#0077BB",
    "gabriel": "#EE7733",
    "delaunay_centroidal": "#009988",
    "voronoi_pruned": "#CC3311"
}

# Marker mapping established in manuscript Figure 5.
MARKERS = {
    "gabriel": "o",
    "delaunay": "s",
    "delaunay_centroidal": "^",
    "voronoi_pruned": "D",
}

# Define legend names with proper capitalization and specific renaming
LEGEND_NAMES = {
    "delaunay": "Delaunay",
    "gabriel": "Gabriel",
    "delaunay_centroidal": "Delaunay-centroidal",
    "voronoi_pruned": "Voronoi"
}

# Define font sizes (matching the reference code with increased size)
FONT_SIZE = 28  # Further increased font size for labels
LEGEND_SIZE = 20  # Further increased size for legend
LINE_WIDTH = 4.0  # Define line thickness for plots

# Define directories for analysis data (assumed to be saved in BASE_DIR/analysis)
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

# Filenames for saved metrics and alpha values
ALPHA_FILE = os.path.join(ANALYSIS_DIR, "alpha_values.npy")
CLASSI_METRICS_FILE = os.path.join(ANALYSIS_DIR, "classI_network_metrics.npy")
RANDOM_METRICS_FILE = os.path.join(ANALYSIS_DIR, "random_network_metrics.npy")

# Load data
if not os.path.exists(ALPHA_FILE):
    raise FileNotFoundError(f"Missing alpha values file: {ALPHA_FILE}")
alpha_values = np.load(ALPHA_FILE)

if not os.path.exists(CLASSI_METRICS_FILE):
    raise FileNotFoundError(f"Missing classI metrics file: {CLASSI_METRICS_FILE}")
metrics_classI = np.load(CLASSI_METRICS_FILE, allow_pickle=True).item()

if not os.path.exists(RANDOM_METRICS_FILE):
    raise FileNotFoundError(f"Missing random metrics file: {RANDOM_METRICS_FILE}")
metrics_random = np.load(RANDOM_METRICS_FILE, allow_pickle=True).item()

# Create the output directory for plots if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_TER_vs_alpha(alpha_vals, metrics_classI, metrics_random):
    """
    For each graph type, plot the normalized TER vs alpha for classI patterns and overlay
    a horizontal dashed line for the TER from random patterns.
    """
    fig, ax = plt.subplots(figsize=plot_cfg["figure_size"])

    # Plot Poisson/random data FIRST (will appear in left column of legend)
    for gtype in GRAPH_TYPES:
        # Plot random TER as horizontal dashed line (assumed constant across alpha)
        ter_random = metrics_random[gtype]["ter"][0]  # one value replicated for all alphas
        ax.axhline(y=ter_random, color=COLORS[gtype], linestyle='--',
                   label=f"Poisson - {LEGEND_NAMES[gtype]}", linewidth=LINE_WIDTH)
        ax.plot(alpha_vals, np.full_like(alpha_vals, ter_random, dtype=float),
                linestyle='None', marker=MARKERS[gtype],
                markersize=plot_cfg["marker_size"],
                markerfacecolor='white', markeredgecolor=COLORS[gtype],
                markeredgewidth=1.5, zorder=3)

    # Invisible spacer: lands at row 5 of the left legend column so that the
    # Hyperuniform entries below align row-for-row with the Poisson entries
    # in column-major fill (with ncol=2). Without this, the 9-entry legend
    # (4 Poisson + 4 Hyperuniform + 1 RSA) puts Hyperuniform-Gabriel into
    # column 1 row 5 and rotates the right column.
    ax.plot([], [], ' ', label=' ')

    # Plot hyperuniform data SECOND (will appear in right column of legend)
    for gtype in GRAPH_TYPES:
        # Plot classI (hyperuniform) TER vs alpha
        ter_classI = np.array(metrics_classI[gtype]["ter"])
        ax.plot(alpha_vals, ter_classI, linestyle='-',
                marker=MARKERS[gtype], color=COLORS[gtype],
                markerfacecolor=COLORS[gtype],
                markeredgecolor=COLORS[gtype],
                markersize=plot_cfg["marker_size"],
                label=f"Hyperuniform - {LEGEND_NAMES[gtype]}",
                linewidth=LINE_WIDTH)

    # Overlay RSA data if available — single high-density (near-saturation) phi.
    rsa_gtype_map = {
        "gabriel": "gabriel", "delaunay": "delaunay",
        "delaunay_centroidal": "delaunay_centroidal",
        "voronoi_pruned": "voronoi"
    }
    if os.path.exists(RSA_DATA_FILE):
        with open(RSA_DATA_FILE) as f:
            rsa_data = json.load(f)
        phi_keys = sorted([k for k in rsa_data if k.startswith("phi_")],
                          key=lambda k: float(k.replace("phi_", "")))
        phi_key = phi_keys[-1]  # highest density
        phi_val = phi_key.replace("phi_", "")
        phi_block = rsa_data[phi_key]
        x_rsa = alpha_vals[-1] + 0.15  # offset to the right
        for gtype in GRAPH_TYPES:
            rsa_key = rsa_gtype_map.get(gtype, gtype)
            entry = phi_block.get(rsa_key)
            if entry and entry.get("ter_mean") is not None:
                ax.scatter([x_rsa], [entry["ter_mean"]],
                           marker='s', color=COLORS[gtype],
                           s=100, linewidths=0, zorder=6)
        # The spacer between the PoPPI and HuPPI handles places this marker in
        # row 5 of the right legend column.
        ax.plot([], [], 's', color='gray', markersize=10,
                label=f'RSA ($\\phi = {phi_val}$)')

    ax.set_xlabel(r'Disorder Strength ($a$)', fontsize=FONT_SIZE, labelpad=15)
    ax.set_ylabel(r'Normalized Total Effective' + '\n' + r'Resistance ($\mathcal{R}_{\text{norm}}$)', fontsize=FONT_SIZE, labelpad=15)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles[:len(GRAPH_TYPES)] = [
        Line2D([0], [0], color=COLORS[gtype], linestyle='--',
               linewidth=LINE_WIDTH, marker=MARKERS[gtype],
               markersize=plot_cfg["marker_size"],
               markerfacecolor='white', markeredgecolor=COLORS[gtype],
               markeredgewidth=1.5)
        for gtype in GRAPH_TYPES
    ]
    ax.legend(legend_handles, legend_labels,
              fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.42, 1.575), ncol=2,
              frameon=False, fancybox=False, shadow=False, handlelength=2.8)
    ax.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)
    ax.set_ylim(0.05, 0.095)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=0.68)

    plot_path = os.path.join(OUTPUT_DIR, "TER_vs_alpha.png")
    plt.savefig(plot_path, dpi=plot_cfg["dpi"])
    plt.close()
    print(f"Plot saved as {plot_path}")

if __name__ == "__main__":
    print("Generating TER vs Alpha plot...")
    plot_TER_vs_alpha(alpha_values, metrics_classI, metrics_random)
    print("Plotting complete!")

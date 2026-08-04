#!/usr/bin/env python3
"""Plot the network metrics computed for manuscript Figure 3.

This script loads the network metrics computed by generate_data.py and generates plots.
For each graph type, it plots the normalized Total Effective Resistance (TER) as a function of lattice size,
comparing HuPPI and PoPPI networks (stored under the internal keys ``classI``
and ``random``, respectively).
The plots are saved to the output directory defined in config.py.

All configuration parameters are read from config.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config

# Load plot parameters from config
plot_params = config.get_plot_params()
BASE_DIR = plot_params["base_dir"]
OUTPUT_DIR = plot_params["output_dir"]
graph_types = plot_params["graph_types"]
colors = plot_params["colors"]
figure_size = plot_params["figure_size"]
dpi = plot_params["dpi"]

ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
metrics_file = os.path.join(ANALYSIS_DIR, "network_metrics.npy")

if not os.path.exists(metrics_file):
    raise FileNotFoundError("Metrics file not found. Run generate_data.py first.")

metrics = np.load(metrics_file, allow_pickle=True).item()

# Create the output directory if necessary
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define directories
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

# Define font sizes (matching the reference code with increased size)
FONT_SIZE = 28  # Further increased font size for labels
LEGEND_SIZE = 20  # Further increased size for legend
LINE_WIDTH = 4.0  # Define line thickness for plots

# Add graph type labels mapping - using consistent naming with fig4
graph_type_labels = {
    "delaunay": "Delaunay",
    "delaunay_centroidal": "Delaunay-centroidal",
    "voronoi": "Voronoi",
    "voronoi_pruned": "Voronoi",  # Keep the 'pruned' distinction as it appears to be a different algorithm
    "gabriel": "Gabriel"
}

# Marker mapping established in manuscript Figure 5.
markers = {
    "gabriel": "o",
    "delaunay": "s",
    "delaunay_centroidal": "^",
    "voronoi_pruned": "D",
}

# Function to compute average number of nodes for a given configuration
def get_avg_nodes(dist, gtype, size, ensemble_size):
    node_counts = []
    for ens_idx in range(ensemble_size):
        base_name = f"{dist}_lsize{size}_ens{ens_idx}_{gtype}"
        edges_path = os.path.join(GRAPHS_DIR, base_name + "_edges.npy")
        edges = np.load(edges_path)
        # Get unique nodes from edges
        num_nodes = len(np.unique(edges[:, :2]))
        node_counts.append(num_nodes)
    return np.mean(node_counts)

# Plot all graph types and distributions on the same axes
plt.figure(figsize=figure_size)
poisson_handles = []
hyperuniform_handles = []

# Plot Poisson/random data FIRST (will appear in left column of legend)
for gtype in graph_types:
    dist = "random"
    data = metrics[dist][gtype]
    sizes = sorted(data.keys())
    # Compute average number of nodes for each lattice size
    num_nodes = [get_avg_nodes(dist, gtype, size, config.get_data_params()["ensemble_size"])
                for size in sizes]
    ter_values = [data[size]["ter"] for size in sizes]
    label = f"Poisson - {graph_type_labels[gtype]}"
    line, = plt.plot(num_nodes, ter_values, linestyle="--",
                     marker=markers[gtype],
                     color=colors[gtype], label=label, linewidth=LINE_WIDTH,
                     markersize=plot_params.get("marker_size", 7),
                     markerfacecolor="white",
                     markeredgecolor=colors[gtype],
                     markeredgewidth=1.5)
    poisson_handles.append(line)

# Plot HuPPI data SECOND (will appear in the right column of the legend)
for gtype in graph_types:
    dist = "classI"
    data = metrics[dist][gtype]
    sizes = sorted(data.keys())
    # Compute average number of nodes for each lattice size
    num_nodes = [get_avg_nodes(dist, gtype, size, config.get_data_params()["ensemble_size"])
                for size in sizes]
    ter_values = [data[size]["ter"] for size in sizes]
    label = f"Hyperuniform - {graph_type_labels[gtype]}"
    line, = plt.plot(num_nodes, ter_values, linestyle="-",
                     marker=markers[gtype],
                     color=colors[gtype], label=label, linewidth=LINE_WIDTH,
                     markersize=plot_params.get("marker_size", 7),
                     markerfacecolor=colors[gtype],
                     markeredgecolor=colors[gtype])
    hyperuniform_handles.append(line)

# Remove title and increase legend font size
plt.xlabel("Number of Nodes ($n$)", fontsize=FONT_SIZE, labelpad=15)
plt.ylabel(r"Normalized Total Effective" + "\n" + r"Resistance ($\mathcal{R}_{\text{norm}}$)", fontsize=FONT_SIZE, labelpad=15)
# plt.grid(True, linestyle='--', alpha=0.7)
legend_handles = poisson_handles + hyperuniform_handles
legend_labels = [handle.get_label() for handle in legend_handles]

plt.legend(legend_handles, legend_labels,
           fontsize=LEGEND_SIZE, loc='upper center',
           bbox_to_anchor=(0.42, 1.42), ncol=2,
           frameon=False, fancybox=False, shadow=False,
           handlelength=2.8, numpoints=1)
plt.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)
plt.ylim(0.05, 0.097)
plt.tight_layout()
plt.subplots_adjust(bottom=0.14, top=0.75)
plot_path = os.path.join(OUTPUT_DIR, "TER_vs_size.png")
plt.savefig(plot_path, dpi=dpi)
plt.close()
print(f"Plot saved as {plot_path}")

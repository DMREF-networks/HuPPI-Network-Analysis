#!/usr/bin/env python3

"""
Creates manuscript Figure 8 from the lattice-size experiment:
- Plots curvature versus network size for HuPPI networks
- Includes all graph types (Delaunay, Gabriel, Delaunay-centroidal, Voronoi)
- Compares with random point patterns
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import get_plot_params

# Get parameters
params = get_plot_params()

# Add default parameters if not in config
params.setdefault("capsize", 3)
params.setdefault("grid_alpha", 0.3)
params.setdefault("hist_bins", 30)
params.setdefault("style1_fig_size", (10, 8))
params.setdefault("dpi", 300)

# Extract parameters
BASE_DIR = params["base_dir"]
OUTPUT_DIR = params["output_dir"]
lattice_size_ranges = params["lattice_size_ranges"]
distribution_classes = params["distribution_classes"]
graph_types = params["graph_types"]
ensemble_size = params["ensemble_size"]
colors = params["colors"]
linestyles = params["linestyles"]
# Define font sizes (matching the reference code with increased size)
FONT_SIZE = 28  # Further increased font size for labels
LEGEND_SIZE = 20  # Further increased size for legend
LINE_WIDTH = 4.0  # Define line thickness for plots

# Add this near the top with other parameter definitions
graph_type_labels = {
    "delaunay": "Delaunay",
    "delaunay_centroidal": "Delaunay-centroidal",
    "voronoi": "Voronoi",
    "voronoi_pruned": "Voronoi",
    "gabriel": "Gabriel"
}

# Marker mapping established in manuscript Figure 5.
markers = {
    "gabriel": "o",
    "delaunay": "s",
    "delaunay_centroidal": "^",
    "voronoi_pruned": "D",
}

def load_hyperuniform_data():
    """Load curvature data for hyperuniform classes"""
    data = {}
    for dist_type in distribution_classes:
        data[dist_type] = {}
        for gtype in graph_types:
            data[dist_type][gtype] = {
                'means': [],
                'stds': [],
                'sizes': [],  # Will store actual node counts from graph files
                'lattice_sizes': list(lattice_size_ranges[gtype])
            }

            for lsize in lattice_size_ranges[gtype]:
                ensemble_curvs = []
                node_counts = []
                for ens_idx in range(ensemble_size):
                    # Load graph nodes to get actual node count
                    nodes_fname = f"{dist_type}_lsize{lsize}_ens{ens_idx}_{gtype}_nodes.npy"
                    nodes_fpath = f"{BASE_DIR}/graphs/{nodes_fname}"

                    # Load curvature data
                    curv_fname = f"{dist_type}_lsize{lsize}_ens{ens_idx}_{gtype}_ollivier.npy"
                    curv_fpath = f"{BASE_DIR}/curvature/{curv_fname}"

                    if os.path.exists(nodes_fpath) and os.path.exists(curv_fpath):
                        nodes = np.load(nodes_fpath)
                        node_counts.append(len(nodes))

                        curv_data = np.load(curv_fpath)
                        ensemble_curvs.append(np.mean(curv_data[:, 2]))

                if ensemble_curvs:
                    data[dist_type][gtype]['means'].append(np.mean(ensemble_curvs))
                    data[dist_type][gtype]['stds'].append(np.std(ensemble_curvs))
                    data[dist_type][gtype]['sizes'].append(np.mean(node_counts))
                else:
                    data[dist_type][gtype]['means'].append(np.nan)
                    data[dist_type][gtype]['stds'].append(np.nan)
                    data[dist_type][gtype]['sizes'].append(np.nan)

    return data

def load_random_data():
    """Load curvature data for random patterns"""
    random_data = {gtype: {'means': [], 'stds': [], 'sizes': [], 'lattice_sizes': []}
                  for gtype in graph_types}

    for gtype in graph_types:
        for lsize in lattice_size_ranges[gtype]:
            ensemble_curvs = []
            node_counts = []

            for ens_idx in range(ensemble_size):
                # Load graph nodes to get actual node count
                nodes_fname = f"random_pattern_lsize{lsize}_ens{ens_idx}_{gtype}_nodes.npy"
                nodes_fpath = f"{BASE_DIR}/graphs/{nodes_fname}"

                # Load curvature data
                curv_fname = f"random_pattern_lsize{lsize}_ens{ens_idx}_{gtype}_ollivier.npy"
                curv_fpath = f"{BASE_DIR}/curvature/{curv_fname}"

                if os.path.exists(nodes_fpath) and os.path.exists(curv_fpath):
                    nodes = np.load(nodes_fpath)
                    node_counts.append(len(nodes))

                    curv_data = np.load(curv_fpath)
                    ensemble_curvs.append(np.mean(curv_data[:, 2]))

            if ensemble_curvs:
                random_data[gtype]['means'].append(np.mean(ensemble_curvs))
                random_data[gtype]['stds'].append(np.std(ensemble_curvs))
                random_data[gtype]['sizes'].append(np.mean(node_counts))
                random_data[gtype]['lattice_sizes'].append(lsize)

    return random_data

def get_display_label(gtype):
    """Convert internal graph type names to display labels"""
    if gtype == 'voronoi_pruned':
        return 'voronoi'
    elif gtype == 'delaunay_centroidal':
        return 'delaunay centroidal'
    return gtype  # Return unchanged for other types

def plot_style_1(hu_data, random_data):
    """Single plot with all graph types and distributions"""
    fig, ax = plt.subplots(figsize=params["style1_fig_size"])
    poisson_handles = []
    hyperuniform_handles = []

    # Plot random and non-classI data (dashed lines) FIRST
    for gtype in graph_types:
        # Plot random data
        handle = ax.errorbar(random_data[gtype]['sizes'],
                             random_data[gtype]['means'],
                             yerr=random_data[gtype]['stds'],
                             color=colors[gtype],
                             linestyle='--',  # dashed line for random
                             marker=markers[gtype],
                             capsize=params["capsize"], markersize=4,
                             markerfacecolor='white',
                             markeredgecolor=colors[gtype],
                             markeredgewidth=1.5,
                             linewidth=LINE_WIDTH,
                             label=f"Poisson - {graph_type_labels[gtype]}")
        poisson_handles.append(handle)

    # Plot hyperuniform data (solid lines) SECOND
    for gtype in graph_types:
        for dist_type in distribution_classes:
            if dist_type == "classI":  # HuPPI data use solid lines
                means = hu_data[dist_type][gtype]['means']
                stds = hu_data[dist_type][gtype]['stds']
                sizes = hu_data[dist_type][gtype]['sizes']

                handle = ax.errorbar(sizes, means, yerr=stds,
                                     color=colors[gtype],
                                     linestyle='-',  # solid line for HuPPI
                                     marker=markers[gtype],
                                     capsize=params["capsize"], markersize=4,
                                     markerfacecolor=colors[gtype],
                                     markeredgecolor=colors[gtype],
                                     linewidth=LINE_WIDTH,
                                     label=f"Hyperuniform - {graph_type_labels[gtype]}")
                hyperuniform_handles.append(handle)

    ax.set_xlabel(r"Number of Nodes ($n$)", fontsize=FONT_SIZE, labelpad=15)
    ax.set_ylabel("Mean ORC", fontsize=FONT_SIZE, labelpad=15)
    legend_handles = poisson_handles + hyperuniform_handles
    legend_labels = [handle.get_label() for handle in legend_handles]

    ax.legend(legend_handles, legend_labels,
              fontsize=LEGEND_SIZE, loc='upper center',
              bbox_to_anchor=(0.5, 1.4), ncol=2,
              frameon=False, fancybox=False, shadow=False,
              handlelength=2.8, numpoints=1, markerscale=2.5)
    ax.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)
    # ax.grid(True, linestyle='--', alpha=params["grid_alpha"])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=0.85)
    plt.savefig(f"{OUTPUT_DIR}/mean_ORC_vs_nodes.png", dpi=params["dpi"], bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading hyperuniform data...")
    hyperuniform_data = load_hyperuniform_data()

    print("Loading random pattern data...")
    random_data = load_random_data()

    print("Generating plots...")
    plot_style_1(hyperuniform_data, random_data)

    print("All done! Generated visualization:")
    print("- mean_ORC_vs_nodes.png - Single plot with all data")

#!/usr/bin/env python3

"""
Create manuscript Figure 7 from the HuPPI disorder sweep and PoPPI baseline.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from config import get_plot_params

# RSA data path
RSA_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "rsa", "rsa_network_properties.json")

# Get parameters
params = get_plot_params()

# Extract parameters
BASE_DIR = params["base_dir"]
OUTPUT_DIR = params["output_dir"]
alpha_values = params["alpha_values"]
distribution_classes = params["distribution_classes"]
graph_types = params["graph_types"]
ensemble_size = params["ensemble_size"]
colors = params["colors"]
linestyles = params["linestyles"]

# Define font sizes (matching the reference code with increased size)
FONT_SIZE = 28  # Further increased font size for labels
LEGEND_SIZE = 21  # Further increased size for legend
LINE_WIDTH = 4.0  # Define line thickness for plots

# Marker mapping established in manuscript Figure 5.
markers = {
    "gabriel": "o",
    "delaunay": "s",
    "delaunay_centroidal": "^",
    "voronoi_pruned": "D",
}

# Add this near the top with other parameter definitions
graph_type_labels = {
    "delaunay": "Delaunay",
    "delaunay_centroidal": "Delaunay-centroidal",
    "voronoi": "Voronoi",
    "voronoi_pruned": "Voronoi",
    "gabriel": "Gabriel"
}

# Define distribution type labels (matching the reference code)
dist_type_labels = {
    "classI": "Hyperuniform",
    "poisson": "Poisson"
}

def load_hyperuniform_data():
    """Load curvature data for hyperuniform classes"""
    data = {}
    for dist_type in distribution_classes:
        data[dist_type] = {}
        for gtype in graph_types:
            data[dist_type][gtype] = {
                'means': [],
                'stds': []
            }

            for alpha in alpha_values:
                ensemble_curvs = []
                for ens_idx in range(ensemble_size):
                    fname = f"{dist_type}_alpha{alpha:.2f}_ens{ens_idx}_{gtype}_ollivier.npy"
                    fpath = f"{BASE_DIR}/curvature/{fname}"
                    if os.path.exists(fpath):
                        curv_data = np.load(fpath)
                        ensemble_curvs.append(np.mean(curv_data[:, 2]))

                if ensemble_curvs:
                    data[dist_type][gtype]['means'].append(np.mean(ensemble_curvs))
                    data[dist_type][gtype]['stds'].append(np.std(ensemble_curvs))
                else:
                    data[dist_type][gtype]['means'].append(np.nan)
                    data[dist_type][gtype]['stds'].append(np.nan)

    return data

def load_random_data():
    """Load the 50-realization PoPPI curvature ensemble."""
    random_data = {gtype: [] for gtype in graph_types}

    for gtype in graph_types:
        for ens_idx in range(ensemble_size):
            fname = f"random_ens{ens_idx}_{gtype}_ollivier.npy"
            fpath = f"{BASE_DIR}/curvature/{fname}"
            if os.path.exists(fpath):
                curv_data = np.load(fpath)
                random_data[gtype].append(np.mean(curv_data[:, 2]))

    return random_data

def plot_style_1(hu_data, random_data):
    """Single plot with all graph types and distributions"""
    # Make figure taller by increasing height
    fig_width, fig_height = params["style1_fig_size"]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot hyperuniform data
    for gtype in graph_types:
        for dist_type in distribution_classes:
            means = hu_data[dist_type][gtype]['means']

            # Update label format to match reference code
            label = f"{dist_type_labels[dist_type]} - {graph_type_labels[gtype]}"

            ax.plot(alpha_values, means, linestyle='-',
                    marker=markers[gtype], color=colors[gtype],
                    markersize=7, markerfacecolor=colors[gtype],
                    markeredgecolor=colors[gtype],
                    linewidth=LINE_WIDTH, label=label)

    # Add random data as horizontal lines
    for gtype in graph_types:
        if random_data[gtype]:
            mean_val = np.mean(random_data[gtype])
            ax.axhline(y=mean_val, color=colors[gtype],
                      linestyle='--', linewidth=LINE_WIDTH,
                      label=f"Poisson - {graph_type_labels[gtype]}")
            ax.plot(alpha_values, np.full(len(alpha_values), mean_val),
                    linestyle='None', marker=markers[gtype], markersize=7,
                    markerfacecolor='white', markeredgecolor=colors[gtype],
                    markeredgewidth=1.5, zorder=3)

    # Overlay RSA data if available — single high-density (near-saturation) phi.
    rsa_gtype_map = {
        "delaunay": "delaunay", "gabriel": "gabriel",
        "delaunay_centroidal": "delaunay_centroidal",
        "voronoi": "voronoi", "voronoi_pruned": "voronoi"
    }
    if os.path.exists(RSA_DATA_FILE):
        with open(RSA_DATA_FILE) as f:
            rsa_data = json.load(f)
        phi_keys = sorted([k for k in rsa_data if k.startswith("phi_")],
                          key=lambda k: float(k.replace("phi_", "")))
        phi_key = phi_keys[-1]
        phi_val = phi_key.replace("phi_", "")
        phi_block = rsa_data[phi_key]
        x_rsa = alpha_values[-1] + 0.15
        for gtype in graph_types:
            rsa_key = rsa_gtype_map.get(gtype, gtype)
            entry = phi_block.get(rsa_key)
            if entry and entry.get("orc_mean") is not None:
                ax.scatter([x_rsa], [entry["orc_mean"]],
                           marker='s', color=colors[gtype],
                           s=100, linewidths=0, zorder=5)
        ax.plot([], [], 's', color='gray', markersize=10,
                label=f'RSA ($\\phi={phi_val}$)')

    ax.set_xlabel(r"Disorder Strength ($a$)", fontsize=FONT_SIZE, labelpad=15)
    ax.set_ylabel("Mean ORC", fontsize=FONT_SIZE, labelpad=15)
    ax.set_ylim(-0.6, 0.2)

    from matplotlib.lines import Line2D
    spacer = Line2D([], [], color='none', label=' ')
    legend_handles = [
        Line2D([0], [0], color=colors[gtype], linestyle='--',
               marker=markers[gtype], markersize=7,
               markerfacecolor='white', markeredgecolor=colors[gtype],
               markeredgewidth=1.5, linewidth=LINE_WIDTH)
        for gtype in graph_types
    ]
    legend_labels = [
        f"Poisson - {graph_type_labels[gtype]}"
        for gtype in graph_types
    ]
    legend_handles.append(spacer)
    legend_labels.append(' ')
    legend_handles.extend(
        Line2D([0], [0], color=colors[gtype], linestyle='-',
               marker=markers[gtype], markersize=7,
               markerfacecolor=colors[gtype],
               markeredgecolor=colors[gtype], linewidth=LINE_WIDTH)
        for gtype in graph_types
    )
    legend_labels.extend(
        f"Hyperuniform - {graph_type_labels[gtype]}"
        for gtype in graph_types
    )
    if os.path.exists(RSA_DATA_FILE):
        legend_handles.append(Line2D([0], [0], color='gray', linestyle='None',
                                     marker='s', markersize=10))
        legend_labels.append(f'RSA ($\\phi={phi_val}$)')

    ax.legend(legend_handles, legend_labels,
              fontsize=LEGEND_SIZE, loc='upper center',
              bbox_to_anchor=(0.5, 1.48), ncol=2,
              frameon=False, fancybox=False, shadow=False,
              handlelength=2.8)
    ax.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=0.82)
    plt.savefig(f"{OUTPUT_DIR}/mean_ORC_vs_alpha.png", dpi=params["dpi"], bbox_inches='tight')
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
    print("- mean_ORC_vs_alpha.png - Single plot with all data")

#!/usr/bin/env python3

"""
plot.py

Loads the Ollivier-Ricci curvature data and produces:
1) Overlapped histograms in a 2 x 2 grid with insets zooming in on the tail (-3 to -1).
   - Titles are shifted slightly up for better spacing.
   - Insets are PRE-styled (clean ticks, readable fonts).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# Enable unicode minus for proper negative signs in ticks
plt.rcParams['axes.unicode_minus'] = True
from config import get_plot_params

###############################################################################
#                     PARAMETERS & DIRECTORY SETUP
###############################################################################
# Get parameters
params = get_plot_params()

# Replace hardcoded values with params.get() calls
BASE_DIR = params["base_dir"]
OUTPUT_DIR = params["output_dir"]
LATTICE_SIZE = params["lattice_size"]
distribution_classes = params["distribution_classes"]
graph_types = params["graph_types"]
dist_to_alpha_map = params["dist_to_alpha_map"]
ensemble_size = params["ensemble_size"]

# Labels
graph_type_labels = {
    'gabriel': 'Gabriel',
    'delaunay': 'Delaunay',
    'delaunay_centroidal': 'Delaunay-centroidal',
    'voronoi_pruned': 'Voronoi'
}

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


###############################################################################
#            HELPER FUNCTION: LOAD & COMBINE OLLIVIER CURVATURE
###############################################################################
def load_ollivier_curvature(dist_type, gtype):
    """
    Loads all ensemble Ollivier-Ricci curvatures for a given dist_type & gtype.
    Returns a 1D numpy array (pooled from all ensemble realizations).
    """
    alpha_val = dist_to_alpha_map[dist_type]
    all_curv = []

    for ens_idx in range(ensemble_size):
        # Filenames follow the convention written by generate_data.py.
        fname_curv = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_{gtype}_ollivier.npy"
        fpath_curv = os.path.join(BASE_DIR, "curvature", fname_curv)
        if not os.path.exists(fpath_curv):
            # Only print warning once per batch to avoid spamming console
            if ens_idx == 0:
                print(f"Warning: File not found (first ensemble): {fpath_curv}")
            continue

        or_data = np.load(fpath_curv)  # shape (M,3): [node1, node2, curvature]
        all_curv.append(or_data[:, 2])

    if len(all_curv) == 0:
        return np.array([])

    # Concatenate curvature values across ensembles
    return np.concatenate(all_curv, axis=0)


###############################################################################
#            1) OVERLAPPED HISTOGRAMS: 4 PANELS (ONE PER GRAPH TYPE)
###############################################################################
def get_data_ranges(target_dists=None):
    """
    Determines global min/max values for curvature and max PROBABILITY MASS
    across the specified datasets.
    """
    all_data_arrays = []
    bin_count = 60

    # 1. Load relevant data
    for gtype in graph_types:
        for dist_type in distribution_classes:
            if target_dists and dist_type not in target_dists:
                continue

            data = load_ollivier_curvature(dist_type, gtype)
            if data.size > 0:
                all_data_arrays.append(data)

    if not all_data_arrays:
        return -2, 1, 1.0  # fallback

    # 2. Determine global min/max from the pooled data
    flat_data = np.concatenate(all_data_arrays)
    min_curv = np.min(flat_data)
    max_curv = np.max(flat_data)

    # 3. Determine max probability mass using common bins
    common_bins = np.linspace(min_curv, max_curv, bin_count + 1)
    max_prob = 0

    for data in all_data_arrays:
        if data.size > 0:
            # Get raw counts
            hist, _ = np.histogram(data, bins=common_bins, density=False)

            # STRICT NORMALIZATION: Divide count by THIS dataset's total size
            prob_hist = hist / data.size
            max_prob = max(max_prob, np.max(prob_hist))

    return min_curv, max_curv, max_prob

def plot_overlapped_histograms():
    """
    Creates a 2x2 figure. Each subplot corresponds to one graph type.
    Shows Probability Mass histograms for HuPPI (classI) and PoPPI (random).
    Includes an inset in the top-left corner zooming in on the tail (-3 to -1).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
    axes = axes.flatten()  # So we can index [0..3]
    bin_count = 60

    # Filter styles: classI = HuPPI (Hyperuniform), random = PoPPI (Poisson)
    styles = {
        'classI': {'label': 'Hyperuniform', 'color': 'blue'},
        'random': {'label': 'Poisson', 'color': 'orange'}
    }

    # Get ranges ONLY for the distributions we are plotting (classI, random)
    min_curv, max_curv, max_prob = get_data_ranges(['classI', 'random'])

    # Define common bins based on these ranges
    common_bins = np.linspace(min_curv, max_curv, bin_count + 1)

    for ax_idx, gtype in enumerate(graph_types):
        ax = axes[ax_idx]

        # === CREATE INSET ===
        # [x, y, width, height] in relative axes coordinates (0 to 1)
        # x=0.16 provides clearance from the main Y-axis labels
        ax_ins = ax.inset_axes([0.16, 0.45, 0.35, 0.35])

        for dist_type in distribution_classes:
            # Filter: Only plot if it's one of the types mentioned in styles
            if dist_type not in styles:
                continue

            data = load_ollivier_curvature(dist_type, gtype)

            if data.size == 0:
                continue

            style = styles[dist_type]

            # === EXPLICIT NORMALIZATION ===
            total_edges = data.size
            weights = np.ones_like(data) / total_edges

            # 1. Plot on MAIN Axis
            ax.hist(data, bins=common_bins, weights=weights, density=False, alpha=0.5,
                    label=style['label'], color=style['color'], edgecolor='k')

            # 2. Plot on INSET Axis (same bins, same weights)
            # Thinner linewidth for the inset for clarity
            ax_ins.hist(data, bins=common_bins, weights=weights, density=False, alpha=0.5,
                        color=style['color'], edgecolor='k', linewidth=0.5)

        # === INSET STYLING (PRE STANDARDS) ===
        ax_ins.set_xlim(-3, -1)
        ax_ins.set_ylim(0, 0.01)

        # Explicit Y-ticks to prevent overcrowding with larger font
        # This keeps the inset clean and legible
        ax_ins.set_yticks([0, 0.005, 0.01])
        ax_ins.set_yticklabels(['0', '0.005', '0.01'])

        # Increase tick label size for legibility (Standard ~12pt for this figure size)
        ax_ins.tick_params(axis='both', which='major', labelsize=12)

        # Add white background to inset so main plot doesn't bleed through
        ax_ins.patch.set_facecolor('white')
        ax_ins.patch.set_alpha(0.9)

        # === MAIN PLOT STYLING ===
        if ax_idx % 2 == 0:
            ax.set_ylabel("Probability Mass", fontsize=20)

        if ax_idx >= 2:
            ax.set_xlabel("Ollivier–Ricci Curvature", fontsize=20)

        ax.set_xlim(min_curv, max_curv)
        ax.set_ylim(0, max_prob * 1.35)  # Extra headroom so the inset and panel label don't crowd the data in (c) Delaunay-centroidal

        # Place subtitle inside the panel (upper-right corner)
        panel_labels = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}
        ax.text(0.97, 0.95, f"{panel_labels[ax_idx]} {graph_type_labels[gtype]}",
                transform=ax.transAxes, fontsize=18, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(False)

    # Single shared legend at the top of the figure, above all four panels —
    # avoids the clash with the inset that loc='best' caused inside panel (a).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fancybox=False, shadow=False,
               handlelength=2.0, handletextpad=0.6)

    plt.tight_layout()

    # Adjust spacing — leave room at the top for the shared legend
    fig.subplots_adjust(top=0.93, wspace=0.15, hspace=0.2)

    # Output filename
    out_path = os.path.join(OUTPUT_DIR, "ORC_distributions.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] Overlapped histograms with insets in {out_path}")


###############################################################################
#            2) SUBPLOT MATRIX (OPTIONAL, KEPT FOR COMPLETENESS)
###############################################################################
def plot_subplots():
    """
    Creates a 4 x 5 grid:
      Rows => 4 graph types
      Columns => 5 distribution types
    """
    fig, axes = plt.subplots(len(graph_types), len(distribution_classes),
                             figsize=(20, 15), sharex=True, sharey=True)
    bin_count = 60

    # Get global ranges for ALL data
    min_curv, max_curv, max_prob = get_data_ranges(None)

    # Define common bins
    common_bins = np.linspace(min_curv, max_curv, bin_count + 1)

    for row_idx, gtype in enumerate(graph_types):
        for col_idx, dist_type in enumerate(distribution_classes):
            ax = axes[row_idx, col_idx]
            data = load_ollivier_curvature(dist_type, gtype)

            if data.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                ax.axis("off")
            else:
                color = 'gray'
                if dist_type == 'classI': color = 'blue'
                if dist_type == 'random': color = 'orange'

                total_edges = data.size
                weights = np.ones_like(data) / total_edges

                ax.hist(data, bins=common_bins, weights=weights, density=False, alpha=0.7,
                        color=color, edgecolor='k')

                ax.set_xlim(min_curv, max_curv)
                ax.set_ylim(0, max_prob * 1.1)

                dist_label = dist_type
                if dist_type == "classI":
                    dist_label = "Hyperuniform"
                elif dist_type == "random":
                    dist_label = "Poisson"

                title_str = f"{dist_label}\n({graph_type_labels[gtype]})"
                ax.set_title(title_str, fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=16)

                if row_idx == 3:
                    ax.set_xlabel("OCR", fontsize=18)
                if col_idx == 0:
                    ax.set_ylabel("Probability Mass", fontsize=18)

                ax.grid(False)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.35, hspace=0.9)
    out_path = os.path.join(OUTPUT_DIR, f"figure1_L{LATTICE_SIZE}_subplots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] Subplot matrix in {out_path}")


###############################################################################
#                            MAIN
###############################################################################
if __name__ == "__main__":
    plot_overlapped_histograms()
    # plot_subplots()

#!/usr/bin/env python3
"""
plot_robustness_separate.py

Loads robustness simulation data and generates a single 2-panel figure (A and B)
combining both removal strategies:
  (a) lowest-curvature-first, (b) random removal.

Colors:
    delaunay: "#0077BB"
    gabriel: "#EE7733"
    delaunay_centroidal: "#009988"
    voronoi: "#CC3311"

Solid lines denote Hyperuniform (Class I) and dashed lines denote Poisson (Random).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import LatticeGenerator, GraphGenerator

# Define font sizes (matching Figure 2)
FONT_SIZE = 28  # Font size for labels
LEGEND_SIZE = 20  # Font size for legend
LINE_WIDTH = 4.0  # Line thickness for plots
MARKER_SIZE = 7

def average_curves(curve_list):
    """
    Given a list of (phis, lcc_fractions) tuples (assumed to be on the same phi grid),
    compute the average lcc_fractions.

    Returns:
      (phis, mean_curve)
    """
    phis = curve_list[0][0]
    curves = np.array([curve[1] for curve in curve_list])
    mean_curve = np.mean(curves, axis=0)
    return phis, mean_curve

def main():
    # Load the simulation results
    data_file = "robustness_data.npy"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found. Run generate_robustness_data.py first.")
    results = np.load(data_file, allow_pickle=True).item()

    # Colors matching fig2 configuration
    colors = {
        "delaunay": "#0077BB",
        "gabriel": "#EE7733",
        "delaunay_centroidal": "#009988",
        "voronoi": "#CC3311"  # Using voronoi color for voronoi_pruned
    }

    # Marker mapping established in manuscript Figure 5.
    markers = {
        "gabriel": "o",
        "delaunay": "s",
        "delaunay_centroidal": "^",
        "voronoi": "D",
    }

    # Graph type labels matching fig2
    graph_type_labels = {
        "delaunay": "Delaunay",
        "gabriel": "Gabriel",
        "delaunay_centroidal": "Delaunay-centroidal",
        "voronoi": "Voronoi"
    }

    # Define line styles for network classes: solid for HuPPI, dashed for PoPPI
    network_styles = {
        "classI": "-",
        "random": "--"
    }

    removal_strategies = ["lowest", "random"]
    GRAPH_TYPES = ["delaunay", "gabriel", "delaunay_centroidal", "voronoi"]

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Build a single 2-panel figure: (a) lowest, (b) random
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panel_info = [
        (axes[0], "random", "(a)", "Random Removal"),
        (axes[1], "lowest", "(b)", "Targeted-by-ORC Removal"),
    ]

    for ax, removal_strategy, panel_label, panel_title in panel_info:
        # Loop order matching Figure 2: outer loop by graph type, inner loop by network class
        # Order graph types to match Figure 2: gabriel, delaunay, delaunay_centroidal, voronoi
        GRAPH_TYPES_ORDERED = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi"]

        for gtype in GRAPH_TYPES_ORDERED:
            for net_class in ["classI", "random"]:
                curve_list = results[removal_strategy][gtype][net_class]
                if not curve_list:
                    continue
                phis, avg_curve = average_curves(curve_list)
                label = f"{'Hyperuniform' if net_class == 'classI' else 'Poisson'} - {graph_type_labels[gtype]}"
                ls = network_styles[net_class]
                ax.plot(phis, avg_curve, linestyle=ls, color=colors[gtype], label=label,
                        alpha=1.0, linewidth=LINE_WIDTH,
                        marker=markers[gtype], markersize=MARKER_SIZE,
                        markerfacecolor=("white" if net_class == "random"
                                         else colors[gtype]),
                        markeredgecolor=colors[gtype],
                        markeredgewidth=(1.5 if net_class == "random" else 1.0),
                        markevery=max(1, len(phis) // 19))

        ax.set_xlabel("Fraction of Edges Removed", fontsize=FONT_SIZE, labelpad=15)
        ax.set_ylabel(r"$\mathrm{LCC}\ \,\mathrm{Size}$", fontsize=FONT_SIZE, labelpad=15)
        ax.set_title(panel_title, fontsize=FONT_SIZE, pad=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=LEGEND_SIZE)
        # Panel label
        ax.text(0.02, 0.90, panel_label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

    # Only place legend on the right panel to reduce clutter
    # Reorder legend handles and labels so Poisson entries are in the left
    # column and Hyperuniform entries are in the right column.
    handles, labels = axes[1].get_legend_handles_labels()

    # Separate Hyperuniform and Poisson entries
    hyperuniform_entries = [(h, l) for h, l in zip(handles, labels) if 'Hyperuniform' in l]
    poisson_entries = [(h, l) for h, l in zip(handles, labels) if 'Poisson' in l]

    # Sort each by graph type order: Gabriel, Delaunay, Delaunay-Centroidal, Voronoi
    graph_order = ["Gabriel", "Delaunay", "Delaunay-centroidal", "Voronoi"]
    hyperuniform_entries.sort(key=lambda x: next((i for i, g in enumerate(graph_order) if g in x[1]), 999))
    poisson_entries.sort(key=lambda x: next((i for i, g in enumerate(graph_order) if g in x[1]), 999))

    reordered_handles = [h for h, l in poisson_entries] + [h for h, l in hyperuniform_entries]
    reordered_labels = [l for h, l in poisson_entries] + [l for h, l in hyperuniform_entries]

    axes[1].legend(
        reordered_handles,
        reordered_labels,
        fontsize=LEGEND_SIZE,
        loc='upper center',
        bbox_to_anchor=(-0.2, 1.55),
        ncol=2,
        frameon=False,
        fancybox=False,
        shadow=False,
        handlelength=2.8,
        handletextpad=0.6,
        borderpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, bottom=0.14, top=0.85)

    out_file = os.path.join(output_dir, "robustness_combined_AB.png")
    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {out_file}")

    # --- Compute and print total edges vs nodes per network type ---
    def compute_edge_node_counts(N=15, dims=2, alpha=0.5, samples=5):
        graph_types = ["delaunay", "gabriel", "delaunay_centroidal", "voronoi"]
        graph_type_labels = {
            "delaunay": "Delaunay",
            "gabriel": "Gabriel",
            "delaunay_centroidal": "Delaunay-Centroidal",
            "voronoi": "Voronoi",
        }
        results_counts = {
            "classI": {g: {"nodes": [], "edges": []} for g in graph_types},
            "random": {g: {"nodes": [], "edges": []} for g in graph_types},
        }

        # Hyperuniform (Class I): perturbed lattice points
        for s in range(samples):
            lat = LatticeGenerator(size=N, dimensions=dims, C=1.0)
            lat.generate_lattice()
            lat.perturb_lattice(alpha=alpha)
            points_hu = lat.get_points()
            box_size = np.array([N, N])
            gen_hu = GraphGenerator(points_hu, box_size)
            for g in graph_types:
                method = getattr(gen_hu, {
                    "delaunay": "periodic_delaunay_tessellation",
                    "gabriel": "periodic_gabriel_graph",
                    "delaunay_centroidal": "periodic_delaunay_centroidal",
                    "voronoi": "periodic_voronoi_tessellation",
                }[g])
                nodes, edges = method()
                results_counts["classI"][g]["nodes"].append(nodes.shape[0])
                results_counts["classI"][g]["edges"].append(edges.shape[0])

        # Poisson (Random): uniform random points
        for s in range(samples):
            num_points = N * N
            points_rand = np.random.uniform(0, N, size=(num_points, dims))
            box_size = np.array([N, N])
            gen_rand = GraphGenerator(points_rand, box_size)
            for g in graph_types:
                method = getattr(gen_rand, {
                    "delaunay": "periodic_delaunay_tessellation",
                    "gabriel": "periodic_gabriel_graph",
                    "delaunay_centroidal": "periodic_delaunay_centroidal",
                    "voronoi": "periodic_voronoi_tessellation",
                }[g])
                nodes, edges = method()
                results_counts["random"][g]["nodes"].append(nodes.shape[0])
                results_counts["random"][g]["edges"].append(edges.shape[0])

        # Summarize
        print("\nTotal edges vs nodes (averaged across samples):")
        for cls in ["classI", "random"]:
            cls_label = "Hyperuniform" if cls == "classI" else "Poisson"
            print(f"  {cls_label}:")
            for g in ["delaunay", "gabriel", "delaunay_centroidal", "voronoi"]:
                n_arr = np.array(results_counts[cls][g]["nodes"]) if results_counts[cls][g]["nodes"] else np.array([np.nan])
                e_arr = np.array(results_counts[cls][g]["edges"]) if results_counts[cls][g]["edges"] else np.array([np.nan])
                n_mean = int(np.round(np.nanmean(n_arr))) if np.isfinite(n_arr).any() else 0
                e_mean = int(np.round(np.nanmean(e_arr))) if np.isfinite(e_arr).any() else 0
                print(f"    - {graph_type_labels[g]:20s} Nodes≈{n_mean:4d}  Edges≈{e_mean:5d}")

    # Run with small sample size to be fast; adjust samples as needed
    compute_edge_node_counts(N=15, dims=2, alpha=0.5, samples=5)

if __name__ == "__main__":
    main()

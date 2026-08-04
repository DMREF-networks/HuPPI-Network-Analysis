#!/usr/bin/env python3
"""
Script to load network data from a pickle file and generate a subplot figure.

The figure is arranged as 2 rows and 4 columns:
    - Top row: Plots for Class I networks.
    - Bottom row: Plots for Random networks.
Each column corresponds to one of the graph types:
    "gabriel", "delaunay", "delaunay_centroidal", and "voronoi".
Each subplot shows a scatter plot of edge-level OR curvature vs effective resistance,
a fitted regression line, and the Pearson correlation coefficient.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pickle_compat import load_numpy_pickle

# --- FONT CONFIGURATION ---
# 1. Reset font family to default (sans-serif) for normal text (titles, ticks, etc.)
plt.rcParams['font.family'] = 'sans-serif'

# 2. Force ONLY the math parts (inside $...$) to use the Computer Modern (LaTeX) font.
plt.rcParams['mathtext.fontset'] = 'cm'
# --------------------------

MARKERS = {
    "gabriel": "o",
    "delaunay": "s",
    "delaunay_centroidal": "^",
    "voronoi": "D",
}

def load_data(file_path):
    """
    Load the data dictionary from the pickle file.
    """
    with open(file_path, "rb") as f:
        data = load_numpy_pickle(f)
    return data

def _average_ranks(values):
    """Average ranks for ties — used by Spearman correlation."""
    m = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.arange(1, m + 1, dtype=float)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = ranks[i:j + 1].mean()
        ranks[i:j + 1] = avg
        i = j + 1
    out = np.empty(m, dtype=float)
    out[order] = ranks
    return out


def compute_panel_correlations(data):
    """Compute per-realization Pearson r and report the mean and std over realizations.

    For each (class, graph_type) panel, computes Pearson r between κ(e) and
    ΔR_tot(e) on each realization separately, then returns the mean and std
    across realizations:
        { class_key: { graph_type: {
            'pearson_r_mean': float, 'pearson_r_std': float,
            'spearman_r_mean': float, 'spearman_r_std': float,
            'n_realizations': int, 'n_edges_mean': float
        } } }
    """
    graph_types = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi"]
    class_keys = ["classI", "random"]

    results = {}
    for data_key in class_keys:
        results[data_key] = {}
        for gtype in graph_types:
            r_values, rho_values, n_edges = [], [], []

            for rec in data.get(data_key, {}).get(gtype, []):
                curvature = rec.get("curvature")
                eff_res = rec.get("effective_resistance")
                if curvature is None or eff_res is None:
                    continue

                x = np.asarray(curvature)[:, 2]
                y = np.asarray(eff_res)
                if x.shape[0] != y.shape[0]:
                    m = min(x.shape[0], y.shape[0])
                    x, y = x[:m], y[:m]

                finite = np.isfinite(x) & np.isfinite(y)
                x_f, y_f = x[finite], y[finite]
                if x_f.size < 2 or np.std(x_f) == 0 or np.std(y_f) == 0:
                    continue

                r_values.append(float(np.corrcoef(x_f, y_f)[0, 1]))
                rx, ry = _average_ranks(x_f), _average_ranks(y_f)
                if np.std(rx) > 0 and np.std(ry) > 0:
                    rho_values.append(float(np.corrcoef(rx, ry)[0, 1]))
                n_edges.append(int(x_f.size))

            if r_values:
                entry = {
                    "pearson_r_mean":  float(np.mean(r_values)),
                    "pearson_r_std":   float(np.std(r_values)),
                    "spearman_r_mean": float(np.mean(rho_values)) if rho_values else float("nan"),
                    "spearman_r_std":  float(np.std(rho_values)) if rho_values else float("nan"),
                    "n_realizations":  len(r_values),
                    "n_edges_mean":    float(np.mean(n_edges)),
                }
            else:
                entry = {
                    "pearson_r_mean":  float("nan"),
                    "pearson_r_std":   float("nan"),
                    "spearman_r_mean": float("nan"),
                    "spearman_r_std":  float("nan"),
                    "n_realizations":  0,
                    "n_edges_mean":    0.0,
                }
            results[data_key][gtype] = entry

    return results

def print_correlations(results):
    """Pretty-print per-realization correlation means and stds."""
    class_display = {"classI": "Hyperuniform", "random": "Poisson"}
    graph_type_display = {
        "delaunay": "Delaunay",
        "gabriel": "Gabriel",
        "delaunay_centroidal": "Delaunay-centroidal",
        "voronoi": "Voronoi",
    }

    print("Correlations (κ vs TER Edge Contribution) — mean ± std over realizations:")
    for cls_key, per_graph in results.items():
        print(f"  {class_display.get(cls_key, cls_key)}:")
        for gtype, stats in per_graph.items():
            label = graph_type_display.get(gtype, gtype)
            r_m = stats.get("pearson_r_mean", float("nan"))
            r_s = stats.get("pearson_r_std", float("nan"))
            rho_m = stats.get("spearman_r_mean", float("nan"))
            rho_s = stats.get("spearman_r_std", float("nan"))
            n_real = stats.get("n_realizations", 0)
            n_edges = stats.get("n_edges_mean", 0)
            r_str = (f"{r_m:.4f} ± {r_s:.4f}" if np.isfinite(r_m) else "nan")
            rho_str = (f"{rho_m:.4f} ± {rho_s:.4f}" if np.isfinite(rho_m) else "nan")
            print(f"    - {label:20s} Pearson r={r_str} | Spearman ρ={rho_str} "
                  f"(n_real={n_real}, n_edges≈{n_edges:.0f})")

def plot_networks(data, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with 2 rows and 4 columns
    nrows, ncols = 2, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))

    # Graph types in the desired order
    graph_types = ["gabriel", "delaunay", "delaunay_centroidal", "voronoi"]
    graph_type_display = {
        "delaunay": "Delaunay",
        "gabriel": "Gabriel",
        "delaunay_centroidal": "Delaunay-centroidal",
        "voronoi": "Voronoi",
    }

    # Map the data keys to display labels
    class_mapping = {
        "classI": "Hyperuniform",
        "random": "Poisson"
    }

    # First pass to find global min/max values — use only realization 0 of each panel,
    # matching the caption which states the scatter shows a single realization per panel.
    all_curvatures = []
    all_eff_res = []
    for data_key in ['classI', 'random']:
        for gtype in graph_types:
            if not data[data_key][gtype]:
                continue
            rec = data[data_key][gtype][0]
            all_curvatures.extend(rec["curvature"][:, 2])
            all_eff_res.extend(rec["effective_resistance"])

    # Compute global limits with some padding
    curv_min, curv_max = np.min(all_curvatures), np.max(all_curvatures)
    eff_res_min, eff_res_max = np.min(all_eff_res), np.max(all_eff_res)

    # Add 5% padding to the limits
    curv_padding = 0.05 * (curv_max - curv_min)
    eff_res_padding = 0.05 * (eff_res_max - eff_res_min)

    xlim = [curv_min - curv_padding, curv_max + curv_padding]
    ylim = [eff_res_min - eff_res_padding, eff_res_max + eff_res_padding]

    # Loop over the two network classes and corresponding rows
    for row_idx, data_key in enumerate(['classI', 'random']):
        for col_idx, gtype in enumerate(graph_types):
            realizations = data[data_key][gtype]
            if not realizations:
                continue
            rec = realizations[0]  # single representative realization per caption
            all_curvatures = np.asarray(rec["curvature"][:, 2])
            all_eff_res = np.asarray(rec["effective_resistance"])

            ax = axs[row_idx, col_idx]
            color = "green" if data_key == "classI" else "blue"

            # Plot scatter data
            ax.scatter(all_curvatures, all_eff_res, color=color,
                       marker=MARKERS[gtype], alpha=0.5, edgecolors='none')
            if row_idx == 0:
                ax.set_title(
                    graph_type_display.get(gtype, gtype.capitalize()),
                    fontsize=24,
                    pad=14,
                )

            # Set consistent axis limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='major', labelsize=18)

    # Add shared x and y labels
    # The text outside $...$ will be normal sans-serif
    # The text inside $...$ will be Computer Modern LaTeX font
    fig.text(0.6, 0.02, r"$\kappa(e)$", ha='center', va='center', fontsize=28)

    fig.tight_layout()
    # Adjust subplot spacing
    # hspace=0.5 spreads the rows out vertically
    # wspace=0.3 spreads the columns out horizontally
    plt.subplots_adjust(left=0.2, bottom=0.12, hspace=0.5, wspace=0.3)

    # Center each row label on the actual vertical span of its subplot row.
    # Deriving these positions after layout keeps the labels aligned if the
    # margins or subplot spacing change.
    for row_idx, data_key in enumerate(['classI', 'random']):
        row_positions = [ax.get_position() for ax in axs[row_idx, :]]
        row_center = (
            min(position.y0 for position in row_positions)
            + max(position.y1 for position in row_positions)
        ) / 2
        fig.text(
            0.15,
            row_center,
            class_mapping[data_key],
            ha='center',
            va='center',
            rotation=90,
            fontsize=32,
        )

    # Center the overall vertical-axis label on the full two-row plot area.
    all_positions = [ax.get_position() for ax in axs.flat]
    plot_center = (
        min(position.y0 for position in all_positions)
        + max(position.y1 for position in all_positions)
    ) / 2
    fig.text(
        0.1,
        plot_center,
        r"TER Edge Contribution ($\Delta \mathcal{R}_{\mathrm{tot}}(e)$)",
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=28,
    )

    output_file = os.path.join(output_dir, "combined_correlation_plots.png")
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Subplot figure saved: {output_file}")

def main():
    data_path = os.path.join("data", "plot7_data", "plot7_data.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    data = load_data(data_path)
    plot_networks(data)
    # Compute and print correlations for all panels
    results = compute_panel_correlations(data)
    print_correlations(results)

if __name__ == "__main__":
    main()

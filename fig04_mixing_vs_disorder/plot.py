#!/usr/bin/env python3
"""Plot the exact worst-start standard-walk result for Figure 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import DATA_OUTPUT_FILE, LOG_PLOT_FILE


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / DATA_OUTPUT_FILE
OUTPUT_FILE = SCRIPT_DIR / "plots" / LOG_PLOT_FILE

GRAPH_TYPES = ["Gabriel", "Delaunay", "Centroidal", "Voronoi"]
COLORS = {
    "Gabriel": "#EE7733",
    "Delaunay": "#0077BB",
    "Centroidal": "#009988",
    "Voronoi": "#CC3311",
}
LABELS = {
    "Gabriel": "Gabriel",
    "Delaunay": "Delaunay",
    "Centroidal": "Delaunay-centroidal",
    "Voronoi": "Voronoi",
}

FONT_SIZE = 28
LEGEND_SIZE = 20
LINE_WIDTH = 4.0
MARKER_SIZE = 7
MARKERS = {
    "Gabriel": "o",
    "Delaunay": "s",
    "Centroidal": "^",
    "Voronoi": "D",
}


def mean(records: list[dict]) -> float:
    """Return the mean only when every realization has a finite result."""
    nonfinite = [
        record["status"]
        for record in records
        if record["status"] != "finite"
    ]
    if nonfinite:
        counts = {
            status: nonfinite.count(status) for status in set(nonfinite)
        }
        raise ValueError(f"the mixing-time dataset is incomplete: {counts}")
    return float(np.mean([record["mixing_time"] for record in records]))


def main() -> None:
    data = json.loads(INPUT_FILE.read_text())
    alpha_values = data["alpha_values"]

    fig, ax = plt.subplots(figsize=(12, 8))
    plotted_values = []

    for graph_type in GRAPH_TYPES:
        poisson_mean = mean(data["random"][graph_type])
        plotted_values.append(poisson_mean)
        ax.axhline(
            poisson_mean,
            color=COLORS[graph_type],
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=f"Poisson - {LABELS[graph_type]}",
        )
        ax.plot(
            alpha_values,
            np.full(len(alpha_values), poisson_mean),
            linestyle="None",
            marker=MARKERS[graph_type],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=COLORS[graph_type],
            markeredgewidth=1.5,
            zorder=3,
        )

    ax.plot([], [], " ", label=" ")

    for graph_type in GRAPH_TYPES:
        huppi_means = [
            mean(data["hyperuniform"][graph_type][str(alpha)])
            for alpha in alpha_values
        ]
        plotted_values.extend(huppi_means)
        ax.plot(
            alpha_values,
            huppi_means,
            linestyle="-",
            marker=MARKERS[graph_type],
            color=COLORS[graph_type],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markerfacecolor=COLORS[graph_type],
            markeredgecolor=COLORS[graph_type],
            label=f"Hyperuniform - {LABELS[graph_type]}",
        )

    if all(data["rsa"].get(graph_type) for graph_type in GRAPH_TYPES):
        rsa_x = alpha_values[-1] + 0.15
        for graph_type in GRAPH_TYPES:
            rsa_mean = mean(data["rsa"][graph_type])
            plotted_values.append(rsa_mean)
            ax.scatter(
                [rsa_x],
                [rsa_mean],
                marker="s",
                color=COLORS[graph_type],
                s=100,
                linewidths=0,
                zorder=5,
            )
        ax.plot(
            [],
            [],
            "s",
            color="gray",
            markersize=10,
            label=f"RSA ($\\phi={data['rsa_phi']}$)",
        )

    ax.set_xlabel(r"Disorder Strength ($a$)", fontsize=FONT_SIZE, labelpad=15)
    ax.set_ylabel(
        "Random-Walk\nMixing Time (log scale)",
        fontsize=FONT_SIZE,
        labelpad=15,
    )
    ax.set_yscale("log")
    ax.set_ylim(
        bottom=min(plotted_values) / 1.5,
        top=max(plotted_values) * 1.8,
    )
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles[:len(GRAPH_TYPES)] = [
        Line2D(
            [0],
            [0],
            color=COLORS[graph_type],
            linestyle="--",
            linewidth=LINE_WIDTH,
            marker=MARKERS[graph_type],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=COLORS[graph_type],
            markeredgewidth=1.5,
        )
        for graph_type in GRAPH_TYPES
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.42, 1.56),
        ncol=2,
        frameon=False,
        fancybox=False,
        shadow=False,
        handlelength=2.8,
    )
    ax.tick_params(axis="both", which="major", labelsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, top=0.70)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

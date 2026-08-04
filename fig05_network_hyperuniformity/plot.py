#!/usr/bin/env python3
"""Plot the hyperuniformity index versus HuPPI disorder strength."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "h_vs_alpha_v1" / "summary.json"
OUTPUT_DIR = ROOT / "plots"
OUTPUT_STEM = OUTPUT_DIR / "network_hyperuniformity_H_vs_alpha"

GRAPH_STYLES = {
    "gabriel": {
        "label": "Gabriel",
        "color": "#EE7733",
        "marker": "o",
    },
    "delaunay": {
        "label": "Delaunay",
        "color": "#0077BB",
        "marker": "s",
    },
    "delaunay_centroidal": {
        "label": "Delaunay-centroidal",
        "color": "#009988",
        "marker": "^",
    },
    "voronoi": {
        "label": "Voronoi",
        "color": "#CC3311",
        "marker": "D",
    },
}

FONT_SIZE = 28
LEGEND_SIZE = 20
LINE_WIDTH = 4.0
MARKER_SIZE = 7
INSET_LABEL_SIZE = 16
INSET_TICK_SIZE = 14


def load_summary(path=DATA_PATH):
    with Path(path).open() as handle:
        return json.load(handle)


def huppi_conditions(summary):
    """Return the HuPPI condition names and disorder strengths in order."""

    pairs = []
    for condition in summary["conditions"]:
        if condition.startswith("huppi_a"):
            pairs.append((float(condition[len("huppi_a"):]), condition))
    pairs.sort()
    if len(pairs) < 2:
        raise ValueError("At least two HuPPI disorder strengths are required")
    return pairs


def plot_H_vs_alpha(summary, output_stem=OUTPUT_STEM):
    """Create the TER-style H-versus-a figure without uncertainty graphics."""

    if "poppi" not in summary["conditions"]:
        raise ValueError("The summary must contain a PoPPI reference ensemble")

    condition_pairs = huppi_conditions(summary)
    a_values = np.asarray([pair[0] for pair in condition_pairs])
    threshold = float(summary["H_threshold"])

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(12, 8))

    poppi_values = {}
    for graph_type, style in GRAPH_STYLES.items():
        huppi_values = np.asarray(
            [
                summary["conditions"][condition][graph_type]["H"]
                for _, condition in condition_pairs
            ],
            dtype=float,
        )
        if np.any(huppi_values <= 0.0):
            raise ValueError(
                f"Nonpositive H estimate cannot be shown on a log axis: "
                f"{graph_type}"
            )
        poppi_value = float(
            summary["conditions"]["poppi"][graph_type]["H"]
        )
        poppi_values[graph_type] = poppi_value

        axis.plot(
            a_values,
            huppi_values,
            color=style["color"],
            linewidth=LINE_WIDTH,
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            zorder=3,
        )
        axis.plot(
            a_values,
            np.full_like(a_values, poppi_value),
            color=style["color"],
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
            zorder=2,
        )

    axis.axhline(
        threshold,
        color="black",
        linewidth=2.0,
        linestyle=":",
        zorder=1,
    )

    axis.set_yscale("log")
    axis.set_ylim(5.0e-6, 1.55)
    axis.set_xlim(0.05, 2.05)
    axis.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0])
    axis.set_xlabel(
        r"Disorder Strength ($a$)", fontsize=FONT_SIZE, labelpad=15
    )
    axis.set_ylabel(
        r"Hyperuniformity Index ($H$)", fontsize=FONT_SIZE, labelpad=15
    )

    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=80)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(
        axis="both", which="major", labelsize=LEGEND_SIZE, direction="out"
    )

    poppi_axis = axis.inset_axes([0.68, 0.12, 0.26, 0.24])
    poppi_min = min(poppi_values.values())
    poppi_max = max(poppi_values.values())
    poppi_pad = 0.12 * (poppi_max - poppi_min)
    for graph_type, style in GRAPH_STYLES.items():
        poppi_value = poppi_values[graph_type]
        poppi_axis.plot(
            [0.135, 0.5, 0.865],
            [poppi_value, poppi_value, poppi_value],
            color=style["color"],
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
        )
    poppi_axis.set_xlim(0.0, 1.0)
    poppi_axis.set_ylim(poppi_min - poppi_pad, poppi_max + poppi_pad)
    poppi_axis.set_xticks([])
    poppi_axis.set_yticks([0.93, 0.95, 0.97])
    poppi_axis.set_ylabel(
        r"$H$", fontsize=INSET_LABEL_SIZE, labelpad=2
    )
    poppi_axis.set_title(
        "Values for PoPPI Networks", fontsize=INSET_LABEL_SIZE, pad=8
    )
    poppi_axis.tick_params(
        axis="y", labelsize=INSET_TICK_SIZE, length=3, pad=2
    )
    poppi_axis.set_facecolor("white")
    for spine in poppi_axis.spines.values():
        spine.set_color("black")

    handles = []
    labels = []
    for graph_type, style in GRAPH_STYLES.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="--",
                linewidth=LINE_WIDTH,
                marker=style["marker"],
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.5,
            )
        )
        labels.append(f"Poisson - {style['label']}")
    for graph_type, style in GRAPH_STYLES.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
            )
        )
        labels.append(f"Hyperuniform - {style['label']}")
    axis.legend(
        handles,
        labels,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.43),
        ncol=2,
        frameon=False,
        fancybox=False,
        shadow=False,
        handlelength=2.8,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.14, top=0.76)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    plot_H_vs_alpha(load_summary())
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")

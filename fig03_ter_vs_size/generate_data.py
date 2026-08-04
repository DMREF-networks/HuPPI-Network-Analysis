#!/usr/bin/env python3
"""Generate the paired HuPPI/PoPPI size sweep for manuscript Figure 3."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import GraphGenerator, LatticeGenerator
from utils.network.measures import Network_Measures
import config


PARAMS = config.get_data_params()
BASE_DIR = Path(PARAMS["base_dir"])
POINTS_DIR = BASE_DIR / "points"
GRAPHS_DIR = BASE_DIR / "graphs"
ANALYSIS_DIR = BASE_DIR / "analysis"

ENSEMBLE_SIZE = PARAMS["ensemble_size"]
DIMENSIONS = PARAMS["dimensions"]
ALPHA = PARAMS["alpha"]
C = PARAMS["C"]
BASE_SEED = PARAMS["base_seed"]
GRAPH_TYPES = PARAMS["graph_types"]
SIZE_RANGES = PARAMS["size_ranges"]
DISTRIBUTIONS = ("classI", "random")


def graph_constructor(generator: GraphGenerator, graph_type: str):
    return {
        "delaunay": generator.periodic_delaunay_tessellation,
        "gabriel": generator.periodic_gabriel_graph,
        "delaunay_centroidal": generator.periodic_delaunay_centroidal,
        "voronoi_pruned": generator.periodic_voronoi_tessellation,
    }[graph_type]


def generate_points(distribution: str, lattice_size: int, seed: int):
    """Generate one reproducible progenitor point pattern."""
    np.random.seed(seed)
    lattice = LatticeGenerator(
        size=lattice_size, dimensions=DIMENSIONS, C=C
    )
    if distribution == "classI":
        lattice.generate_lattice()
        lattice.perturb_lattice(alpha=ALPHA)
        return lattice.get_points()
    return lattice.generate_random()


def realization_seed(lattice_size: int, ensemble_index: int, population_index: int):
    sequence = np.random.SeedSequence(
        [BASE_SEED, lattice_size, ensemble_index, population_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def initialize_metrics():
    return {
        distribution: {
            graph_type: {
                lattice_size: {"ter": [], "edge_sum": []}
                for lattice_size in SIZE_RANGES[graph_type]
            }
            for graph_type in GRAPH_TYPES
        }
        for distribution in DISTRIBUTIONS
    }


def main():
    for directory in (POINTS_DIR, GRAPHS_DIR, ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    metrics = initialize_metrics()
    measures = Network_Measures()
    lattice_sizes = sorted(
        {size for sizes in SIZE_RANGES.values() for size in sizes}
    )

    print(
        f"Figure 3 generation: a={ALPHA}, {ENSEMBLE_SIZE} realizations "
        f"per population and size"
    )
    for lattice_size in lattice_sizes:
        active_graph_types = [
            graph_type
            for graph_type in GRAPH_TYPES
            if lattice_size in SIZE_RANGES[graph_type]
        ]
        print(
            f"  lattice size {lattice_size}: "
            f"{', '.join(active_graph_types)}",
            flush=True,
        )
        box_dimensions = np.array([lattice_size, lattice_size])

        for ensemble_index in range(ENSEMBLE_SIZE):
            for population_index, distribution in enumerate(DISTRIBUTIONS):
                seed = realization_seed(
                    lattice_size, ensemble_index, population_index
                )
                points = generate_points(distribution, lattice_size, seed)
                point_name = (
                    f"{distribution}_lsize{lattice_size}_ens{ensemble_index}"
                    "_points.npy"
                )
                np.save(POINTS_DIR / point_name, points)

                generator = GraphGenerator(points, box_dimensions)
                for graph_type in active_graph_types:
                    nodes, edges = graph_constructor(generator, graph_type)()
                    base_name = (
                        f"{distribution}_lsize{lattice_size}_ens{ensemble_index}_"
                        f"{graph_type}"
                    )
                    np.save(GRAPHS_DIR / f"{base_name}_nodes.npy", nodes)
                    np.save(GRAPHS_DIR / f"{base_name}_edges.npy", edges)

                    ter = measures.compute_effective_resistance(edges)
                    node_count = len(nodes)
                    ter_norm = ter / (
                        node_count**2 * np.log(node_count)
                    )
                    record = metrics[distribution][graph_type][lattice_size]
                    record["ter"].append(ter_norm)
                    record["edge_sum"].append(float(np.sum(edges[:, 2])))

    for distribution in DISTRIBUTIONS:
        for graph_type in GRAPH_TYPES:
            for lattice_size, record in metrics[distribution][graph_type].items():
                for measure_name, values in record.items():
                    record[measure_name] = float(np.mean(values))

    output = ANALYSIS_DIR / "network_metrics.npy"
    np.save(output, metrics)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

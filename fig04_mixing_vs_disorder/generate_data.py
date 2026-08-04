#!/usr/bin/env python3
"""Generate exact worst-start standard-walk mixing times for Figure 4.

The three non-Voronoi constructions use 100 realizations per condition. The
heavy-tailed Voronoi calculation uses 1,000 realizations per condition. The
stored 100-realization RSA graph ensemble is reprocessed with the same
mixing-time definition.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

# Each process performs dense eigendecompositions. Limit the numerical backend
# to one thread so that process-level parallelism does not oversubscribe CPUs.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ALPHA_VALUES,
    BASE_SEED,
    DATA_OUTPUT_FILE,
    DEFAULT_WORKERS,
    ENSEMBLE_SIZE,
    EPSILON,
    LATTICE_SIZE,
    VORONOI_ENSEMBLE_SIZE,
)
from utils.network.generators import GraphGenerator, LatticeGenerator  # noqa: E402
from utils.network.measures import Network_Measures  # noqa: E402


OUTPUT_FILE = SCRIPT_DIR / DATA_OUTPUT_FILE
DATA_ROOT = Path(os.environ.get("HUPPI_DATA_ROOT", ROOT))
RSA_GRAPH_DIR = DATA_ROOT / "data" / "rsa" / "graphs"
RSA_POINTS_DIR = DATA_ROOT / "data" / "rsa" / "points"

GRAPH_TYPES = ["Gabriel", "Delaunay", "Centroidal", "Voronoi"]
GRAPH_METHODS = {
    "Gabriel": "periodic_gabriel_graph",
    "Delaunay": "periodic_delaunay_tessellation",
    "Centroidal": "periodic_delaunay_centroidal",
    "Voronoi": "periodic_voronoi_tessellation",
}
RSA_FILE_LABELS = {
    "Gabriel": "gabriel",
    "Delaunay": "delaunay",
    "Centroidal": "delaunay_centroidal",
    "Voronoi": "voronoi",
}


def mixing_time_record(
    measures: Network_Measures, edges: np.ndarray, n_nodes: int
) -> dict[str, int | str | None]:
    """Return a mixing time together with an explicit convergence status."""
    try:
        mixing_time = measures.compute_mixing_time_weighted_worst_case_exact(
            edges, n_nodes, epsilon=EPSILON
        )
    except RuntimeError as error:
        message = str(error)
        if "periodic" in message:
            return {"mixing_time": None, "status": "periodic"}
        if "maximum" in message:
            return {"mixing_time": None, "status": "beyond_max_steps"}
        raise
    return {"mixing_time": mixing_time, "status": "finite"}


def mixing_times_for_points(
    points: np.ndarray, graph_types: tuple[str, ...]
) -> dict[str, dict[str, int | str | None]]:
    graph_generator = GraphGenerator(
        points, np.array([LATTICE_SIZE, LATTICE_SIZE])
    )
    measures = Network_Measures()
    result = {}
    for graph_type in graph_types:
        nodes, edges = getattr(
            graph_generator, GRAPH_METHODS[graph_type]
        )()
        result[graph_type] = mixing_time_record(measures, edges, len(nodes))
    return result


def generate_pattern_task(task: tuple) -> tuple:
    population, alpha, ensemble_index, seed, graph_types = task
    np.random.seed(seed)

    if population == "random":
        points = np.random.uniform(
            0, LATTICE_SIZE, size=(LATTICE_SIZE**2, 2)
        )
    else:
        lattice = LatticeGenerator(size=LATTICE_SIZE, dimensions=2, C=1.0)
        lattice.generate_lattice()
        lattice.perturb_lattice(alpha=alpha)
        points = lattice.get_points()

    return (
        population,
        alpha,
        ensemble_index,
        mixing_times_for_points(points, graph_types),
    )


def rsa_task(
    ensemble_index: int,
) -> tuple[int, dict[str, dict[str, int | str | None]]]:
    points = np.load(
        RSA_POINTS_DIR / f"rsa_phi0.53_ens{ensemble_index}_points.npy"
    )
    expected_shape = (LATTICE_SIZE**2, 2)
    if points.shape != expected_shape or not np.isfinite(points).all():
        raise ValueError(
            f"RSA realization {ensemble_index} must have shape {expected_shape} "
            "and contain only finite coordinates"
        )

    measures = Network_Measures()
    result = {}
    for graph_type, file_label in RSA_FILE_LABELS.items():
        nodes = np.load(
            RSA_GRAPH_DIR
            / f"rsa_phi0.53_ens{ensemble_index}_{file_label}_nodes.npy"
        )
        edges = np.load(
            RSA_GRAPH_DIR
            / f"rsa_phi0.53_ens{ensemble_index}_{file_label}_edges.npy"
        )
        expected_nodes = (
            LATTICE_SIZE**2
            if graph_type in ("Gabriel", "Delaunay")
            else 2 * LATTICE_SIZE**2
        )
        if len(nodes) != expected_nodes:
            raise ValueError(
                f"RSA realization {ensemble_index} {graph_type} graph has "
                f"{len(nodes)} nodes; expected {expected_nodes}. Regenerate the "
                "RSA graphs before reprocessing mixing times."
            )
        result[graph_type] = mixing_time_record(measures, edges, len(nodes))
    return ensemble_index, result


def calculate_rsa_ensemble(workers: int) -> dict[str, list[dict]]:
    """Reprocess the corrected 100-realization RSA graph ensemble."""
    rsa_data = {graph_type: [None] * 100 for graph_type in GRAPH_TYPES}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(rsa_task, index) for index in range(100)]
        for future in as_completed(futures):
            ensemble_index, result = future.result()
            for graph_type, record in result.items():
                rsa_data[graph_type][ensemble_index] = record

    missing = {
        graph_type: [
            index for index, record in enumerate(records) if record is None
        ]
        for graph_type, records in rsa_data.items()
        if any(record is None for record in records)
    }
    if missing:
        raise RuntimeError(f"missing RSA mixing-time results: {missing}")
    return rsa_data


def write_json_atomic(path: Path, data: dict) -> None:
    """Replace a JSON output only after its complete contents are serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_tasks(
    alpha_values: list[float],
    ensemble_size: int,
    voronoi_ensemble_size: int,
    seed: int,
) -> list[tuple]:
    seed_sequences = np.random.SeedSequence(seed).spawn(len(alpha_values) + 1)
    tasks = []
    all_graph_types = tuple(GRAPH_TYPES)
    voronoi_only = ("Voronoi",)

    random_seeds = seed_sequences[0].generate_state(voronoi_ensemble_size)
    for ensemble_index in range(ensemble_size):
        tasks.append(
            (
                "random",
                None,
                ensemble_index,
                int(random_seeds[ensemble_index]),
                all_graph_types,
            )
        )
    for ensemble_index in range(ensemble_size, voronoi_ensemble_size):
        tasks.append(
            (
                "random",
                None,
                ensemble_index,
                int(random_seeds[ensemble_index]),
                voronoi_only,
            )
        )

    for alpha, seed_sequence in zip(alpha_values, seed_sequences[1:]):
        alpha_seeds = seed_sequence.generate_state(voronoi_ensemble_size)
        for ensemble_index in range(ensemble_size):
            tasks.append(
                (
                    "hyperuniform",
                    alpha,
                    ensemble_index,
                    int(alpha_seeds[ensemble_index]),
                    all_graph_types,
                )
            )
        for ensemble_index in range(ensemble_size, voronoi_ensemble_size):
            tasks.append(
                (
                    "hyperuniform",
                    alpha,
                    ensemble_index,
                    int(alpha_seeds[ensemble_index]),
                    voronoi_only,
                )
            )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument(
        "--voronoi-ensemble-size",
        type=int,
        default=VORONOI_ENSEMBLE_SIZE,
        help="HuPPI/PoPPI sample count for Voronoi networks only",
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--alpha-values", type=float, nargs="+", default=ALPHA_VALUES
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    rsa_options = parser.add_mutually_exclusive_group()
    rsa_options.add_argument(
        "--skip-rsa",
        action="store_true",
        help="omit the stored 100-realization RSA ensemble",
    )
    rsa_options.add_argument(
        "--rsa-only",
        action="store_true",
        help=(
            "replace only the RSA values in an existing exact-mixing output; "
            "use after regenerating the fixed-count RSA graph ensemble"
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.voronoi_ensemble_size < args.ensemble_size:
        parser.error(
            "--voronoi-ensemble-size cannot be smaller than --ensemble-size"
        )

    if args.rsa_only:
        if not args.output.is_file():
            parser.error(
                f"--rsa-only requires an existing exact-mixing file: {args.output}"
            )
        data = json.loads(args.output.read_text())
        required = {"metadata", "alpha_values", "hyperuniform", "random"}
        missing = sorted(required.difference(data))
        if missing:
            parser.error(
                f"existing output is missing required sections: {missing}"
            )
        metadata = data["metadata"]
        if (
            metadata.get("initial_state") != "worst starting node"
            or metadata.get("epsilon") != EPSILON
        ):
            parser.error(
                "existing output does not use the authoritative exact "
                "worst-start mixing-time definition"
            )

        started = time.perf_counter()
        print("Reprocessing the 100 corrected RSA realizations...", flush=True)
        data["rsa"] = calculate_rsa_ensemble(args.workers)
        data["rsa_phi"] = 0.53
        data["metadata"]["rsa_ensemble_size"] = 100
        write_json_atomic(args.output, data)
        elapsed = time.perf_counter() - started
        print(f"Updated only RSA data in {args.output} after {elapsed:.1f}s")
        return

    alpha_values = [round(value, 10) for value in args.alpha_values]
    data = {
        "metadata": {
            "walk": "weighted standard random walk",
            "initial_state": "worst starting node",
            "epsilon": EPSILON,
            "numerical_method": (
                "reversible symmetric eigendecomposition with doubling and "
                "binary search"
            ),
            "lattice_size": LATTICE_SIZE,
            "ensemble_size": args.ensemble_size,
            "ensemble_size_by_graph": {
                graph_type: (
                    args.voronoi_ensemble_size
                    if graph_type == "Voronoi"
                    else args.ensemble_size
                )
                for graph_type in GRAPH_TYPES
            },
            "base_seed": args.base_seed,
        },
        "alpha_values": alpha_values,
        "hyperuniform": {
            graph_type: {
                str(alpha): [None]
                * (
                    args.voronoi_ensemble_size
                    if graph_type == "Voronoi"
                    else args.ensemble_size
                )
                for alpha in alpha_values
            }
            for graph_type in GRAPH_TYPES
        },
        "random": {
            graph_type: [None]
            * (
                args.voronoi_ensemble_size
                if graph_type == "Voronoi"
                else args.ensemble_size
            )
            for graph_type in GRAPH_TYPES
        },
        "rsa": {graph_type: [] for graph_type in GRAPH_TYPES},
        "rsa_phi": 0.53,
    }

    tasks = build_tasks(
        alpha_values,
        args.ensemble_size,
        args.voronoi_ensemble_size,
        args.base_seed,
    )
    started = time.perf_counter()
    print(
        f"Calculating {len(tasks)} HuPPI/PoPPI patterns with "
        f"{args.workers} workers...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_pattern_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            population, alpha, ensemble_index, result = future.result()
            for graph_type, record in result.items():
                if population == "random":
                    data["random"][graph_type][ensemble_index] = record
                else:
                    data["hyperuniform"][graph_type][str(alpha)][
                        ensemble_index
                    ] = record

            if completed % 100 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                remaining = (len(futures) - completed) / rate
                print(
                    f"  {completed}/{len(futures)} complete; "
                    f"elapsed {elapsed:.1f}s, ETA {remaining:.1f}s",
                    flush=True,
                )

    write_json_atomic(args.output, data)
    print(f"Saved pre-RSA checkpoint to {args.output}", flush=True)

    if not args.skip_rsa:
        print("Calculating the 100 stored RSA realizations...", flush=True)
        data["rsa"] = calculate_rsa_ensemble(args.workers)
        data["metadata"]["rsa_ensemble_size"] = 100

    write_json_atomic(args.output, data)
    elapsed = time.perf_counter() - started
    print(f"Saved {args.output} after {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

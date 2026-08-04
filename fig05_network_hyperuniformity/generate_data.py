#!/usr/bin/env python3
"""Regenerate the paired network spectra for manuscript Figure 5.

The important methodological choices are:

* one progenitor point pattern is shared by all four network constructions;
* the beam width is set by the progenitor point density, not the number of
  nodes in the derived graph;
* spectra are averaged across the ensemble before H is evaluated;
* chi_V(0) is the intercept of a cubic fit to the ensemble low-k spectrum;
* per-realization spectra are retained so the complete estimator can be
  bootstrapped without repeating the expensive rasterization and FFT steps.

The cubic intercept follows the explicit procedure used by Chen, Lomba, and
Torquato (PCCP 20, 17557, 2018).  Because low-k extrapolations are sensitive to
the chosen window, the primary fit ends at k/(2*pi*sqrt(rho)) = 0.30 and nearby
windows are recorded as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import fft
from skimage.draw import polygon


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "data" / "h_vs_alpha_v1"

LATTICE_SIZE = 60
BOX_LENGTH = float(LATTICE_SIZE)
DIMENSION = 2
POINT_DENSITY = LATTICE_SIZE**2 / BOX_LENGTH**2
CHARACTERISTIC_K = 2.0 * np.pi * np.sqrt(POINT_DENSITY)
BEAM_HALF_WIDTH = (1.0 / 20.0) * POINT_DENSITY ** (-1.0 / DIMENSION)
DEFAULT_NPIX = 4096
DEFAULT_ENSEMBLE_SIZE = 200
DEFAULT_WORKERS = min(32, os.cpu_count() or 1)
FFT_WORKERS_PER_PROCESS = 2
BIN_WIDTH_IN_KMIN = 0.5
MAX_RADIAL_MODE = 512
BASE_SEED = 20260713
BOOTSTRAP_REPLICATES = 2000
PRIMARY_FIT_FRACTION = 0.30
FIT_FRACTIONS = (0.20, 0.25, 0.30, 0.35, 0.40)
H_THRESHOLD = 1.0e-2

CONDITIONS = {
    "huppi_a0.1": {"kind": "url", "alpha": 0.1},
    "huppi_a1.5": {"kind": "url", "alpha": 1.5},
    "poppi": {"kind": "poisson", "alpha": None},
}
DEFAULT_CONDITIONS = tuple(
    [f"huppi_a{alpha:.1f}" for alpha in np.arange(0.1, 2.01, 0.1)]
    + ["poppi"]
)
LEGACY_SEED_INDICES = {"huppi_a0.1": 0, "huppi_a1.5": 1, "poppi": 2}
HU_CONDITION_PATTERN = re.compile(
    r"huppi_a(?P<alpha>(?:0|[1-9]\d*)(?:\.\d+)?)\Z"
)

GRAPH_TYPES = (
    "gabriel",
    "delaunay",
    "delaunay_centroidal",
    "voronoi",
)


def _load_generators_without_utils_side_effects():
    """Load generators.py without executing the broad utils/__init__.py.

    The repository package initializer imports optional analysis stacks that
    are unrelated to this figure and are incompatible with the active NumPy
    environment.  Constructing the two lightweight package objects here still
    permits the relative import used by generators.py.
    """

    utils_path = REPO_ROOT / "utils"
    network_path = utils_path / "network"

    if "utils" not in sys.modules:
        package = types.ModuleType("utils")
        package.__path__ = [str(utils_path)]
        sys.modules["utils"] = package
    if "utils.network" not in sys.modules:
        package = types.ModuleType("utils.network")
        package.__path__ = [str(network_path)]
        sys.modules["utils.network"] = package

    module_name = "utils.network.generators"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name, network_path / "generators.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load utils/network/generators.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_generators = _load_generators_without_utils_side_effects()
LatticeGenerator = _generators.LatticeGenerator
GraphGenerator = _generators.GraphGenerator


def realization_seed(condition_index: int, ensemble_index: int) -> int:
    """Return a stable seed independent of the selected ensemble subset."""

    sequence = np.random.SeedSequence(
        [BASE_SEED, int(condition_index), int(ensemble_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def condition_settings(condition: str):
    """Return settings for a named default or arbitrary HuPPI condition."""

    if condition in CONDITIONS:
        return CONDITIONS[condition]
    match = HU_CONDITION_PATTERN.fullmatch(condition)
    if match is None:
        raise ValueError(
            f"Unknown condition {condition!r}; expected 'poppi' or "
            "'huppi_a<nonnegative number>'"
        )
    return {"kind": "url", "alpha": float(match.group("alpha"))}


def condition_seed_index(condition: str) -> int:
    """Assign stable seed streams that reproduce the existing Figure 5 data."""

    if condition in LEGACY_SEED_INDICES:
        return LEGACY_SEED_INDICES[condition]
    alpha = condition_settings(condition)["alpha"]
    return 1000 + int(round(1000.0 * alpha))


def generate_points(condition: str, seed: int):
    """Generate one density-one progenitor point pattern."""

    settings = condition_settings(condition)
    np.random.seed(seed)
    lattice = LatticeGenerator(LATTICE_SIZE, dimensions=2, C=1)
    if settings["kind"] == "url":
        lattice.generate_lattice()
        lattice.perturb_lattice(settings["alpha"])
        points = lattice.points
    else:
        points = lattice.generate_random()
    return np.asarray(points), np.asarray(lattice.box_size)


def generate_network(points, box_size, graph_type: str):
    """Construct one periodic network using the repository implementation."""

    generator = GraphGenerator(points, box_size)
    dispatch = {
        "gabriel": generator.periodic_gabriel_graph,
        "delaunay": generator.periodic_delaunay_tessellation,
        "delaunay_centroidal": generator.periodic_delaunay_centroidal,
        "voronoi": generator.periodic_voronoi_tessellation,
    }
    nodes, edges = dispatch[graph_type]()
    return np.asarray(nodes), np.asarray(edges)


def rasterize_network(nodes, edges, box_length: float, delta: float, npix: int):
    """Rasterize periodic thickened edges by testing pixel centers.

    Each edge becomes the rectangle specified in the manuscript: full width
    2*delta and length b + 2*delta.  ``skimage.draw.polygon`` on coordinates
    shifted by half a pixel exactly reproduces the existing center-sampling
    rasterizer while being much faster at manuscript-quality resolutions.
    """

    dx = box_length / npix
    image = np.zeros((npix, npix), dtype=np.uint8)

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        p1 = nodes[i]
        displacement = nodes[j] - p1
        displacement -= box_length * np.round(displacement / box_length)
        length = np.linalg.norm(displacement)
        if length < 1.0e-12:
            continue

        tangent = displacement / length
        normal = np.array([-tangent[1], tangent[0]])
        p2 = p1 + displacement
        corners = np.array(
            [
                p1 - delta * tangent - delta * normal,
                p1 - delta * tangent + delta * normal,
                p2 + delta * tangent + delta * normal,
                p2 + delta * tangent - delta * normal,
            ]
        )

        lower = corners.min(axis=0)
        upper = corners.max(axis=0)
        x_shifts = [
            shift
            for shift in (-box_length, 0.0, box_length)
            if upper[0] + shift >= 0.0 and lower[0] + shift <= box_length
        ]
        y_shifts = [
            shift
            for shift in (-box_length, 0.0, box_length)
            if upper[1] + shift >= 0.0 and lower[1] + shift <= box_length
        ]

        for x_shift in x_shifts:
            for y_shift in y_shifts:
                shifted = corners + np.array([x_shift, y_shift])
                pixel_vertices = shifted / dx - 0.5
                rows, columns = polygon(
                    pixel_vertices[:, 1],
                    pixel_vertices[:, 0],
                    shape=image.shape,
                )
                image[rows, columns] = 1

    return image


_BIN_CACHE = {}


def radial_bin_layout(npix: int, box_length: float, max_radial_mode: int):
    """Precompute the radial bins needed from an rFFT half-plane."""

    key = (int(npix), float(box_length), int(max_radial_mode))
    if key in _BIN_CACHE:
        return _BIN_CACHE[key]

    radial_limit = min(max_radial_mode, npix // 2 - 1)
    ky = np.concatenate(
        [
            np.arange(0, radial_limit + 1, dtype=np.float64),
            np.arange(-radial_limit, 0, dtype=np.float64),
        ]
    )
    kx = np.arange(0, radial_limit + 1, dtype=np.float64)
    radii = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    valid = (radii > 0.0) & (radii <= radial_limit)

    # The omitted negative-kx half-plane has the same radial total.  Doubling
    # kx>0 modes therefore gives the full-plane angular average.
    x_weights = np.where(kx == 0.0, 1.0, 2.0)
    weights = np.broadcast_to(x_weights[None, :], radii.shape)
    bin_indices = np.floor(
        radii / BIN_WIDTH_IN_KMIN + 1.0e-12
    ).astype(np.int32)

    flat_bins = bin_indices[valid]
    flat_weights = weights[valid]
    counts = np.bincount(flat_bins, weights=flat_weights)
    radius_sums = np.bincount(
        flat_bins, weights=(radii * weights)[valid]
    )
    populated = counts > 0.0
    k_values = (
        radius_sums[populated]
        / counts[populated]
        * (2.0 * np.pi / box_length)
    )

    # Maher and Newhall include the square-pixel form factor.
    pixel_form_factor_sq = (
        np.sinc(ky[:, None] / npix) * np.sinc(kx[None, :] / npix)
    ) ** 2

    layout = {
        "radial_limit": radial_limit,
        "valid": valid,
        "bin_indices": bin_indices,
        "weights": weights,
        "counts": counts,
        "populated": populated,
        "k_values": k_values,
        "pixel_form_factor_sq": pixel_form_factor_sq,
    }
    _BIN_CACHE[key] = layout
    return layout


def spectral_density(image, box_length: float, max_radial_mode: int):
    """Return an angularly averaged spectral density from a binary image."""

    npix = image.shape[0]
    dx = box_length / npix
    field = image.astype(np.float32)
    field -= field.mean()
    transform = fft.rfft2(field, workers=FFT_WORKERS_PER_PROCESS)

    layout = radial_bin_layout(npix, box_length, max_radial_mode)
    radial_limit = layout["radial_limit"]
    selected = np.concatenate(
        [
            transform[: radial_limit + 1, : radial_limit + 1],
            transform[-radial_limit:, : radial_limit + 1],
        ],
        axis=0,
    )
    power = selected.real**2 + selected.imag**2
    power *= layout["pixel_form_factor_sq"]
    power *= dx**4 / box_length**2

    valid = layout["valid"]
    sums = np.bincount(
        layout["bin_indices"][valid],
        weights=(power * layout["weights"])[valid],
        minlength=len(layout["counts"]),
    )
    chi_values = (
        sums[layout["populated"]] / layout["counts"][layout["populated"]]
    )
    return layout["k_values"], chi_values


def _safe_condition_name(condition: str) -> str:
    return condition.replace(".", "p")


def run_realization(task):
    """Generate and analyze all four networks for one progenitor pattern."""

    (
        condition,
        condition_index,
        ensemble_index,
        seed,
        npix,
        max_radial_mode,
        raw_root,
    ) = task
    started = time.time()
    try:
        points, box_size = generate_points(condition, seed)
        spectra = []
        phase_fractions = []

        if raw_root is not None:
            condition_dir = Path(raw_root) / _safe_condition_name(condition)
            points_dir = condition_dir / "points"
            graphs_dir = condition_dir / "graphs"
            points_dir.mkdir(parents=True, exist_ok=True)
            graphs_dir.mkdir(parents=True, exist_ok=True)
            np.save(points_dir / f"ens{ensemble_index:03d}_points.npy", points)

        for graph_type in GRAPH_TYPES:
            nodes, edges = generate_network(points, box_size, graph_type)
            if len(edges) == 0:
                raise RuntimeError(f"{graph_type} produced no edges")

            if raw_root is not None:
                base = graphs_dir / f"ens{ensemble_index:03d}_{graph_type}"
                np.save(f"{base}_nodes.npy", nodes)
                np.save(f"{base}_edges.npy", edges)

            image = rasterize_network(
                nodes,
                edges,
                BOX_LENGTH,
                BEAM_HALF_WIDTH,
                npix,
            )
            k_values, chi_values = spectral_density(
                image, BOX_LENGTH, max_radial_mode
            )
            spectra.append(chi_values)
            phase_fractions.append(float(image.mean()))

        return {
            "condition": condition,
            "condition_index": condition_index,
            "ensemble_index": ensemble_index,
            "seed": seed,
            "k_values": k_values,
            "spectra": np.asarray(spectra),
            "phase_fractions": np.asarray(phase_fractions),
            "elapsed_seconds": time.time() - started,
        }
    except Exception as error:
        return {
            "condition": condition,
            "condition_index": condition_index,
            "ensemble_index": ensemble_index,
            "seed": seed,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.time() - started,
        }


def cubic_intercept_H(k_values, chi_values, fit_fraction: float):
    """Fit a cubic low-k polynomial and return its k=0 intercept and H."""

    peak_index = int(np.nanargmax(chi_values))
    peak_value = float(chi_values[peak_index])
    peak_k = float(k_values[peak_index])
    if peak_value <= 0.0 or peak_k <= 0.0:
        raise ValueError("Spectrum has no positive peak")

    fit_k_max = fit_fraction * CHARACTERISTIC_K
    use = (k_values > 0.0) & (k_values <= fit_k_max)
    if np.count_nonzero(use) < 6:
        raise ValueError("Fewer than six populated bins in the fit window")

    # Scaling k by the fit endpoint improves conditioning without changing the
    # intercept.  Coefficients are in ascending polynomial order.
    scaled_k = k_values[use] / fit_k_max
    coefficients = np.polynomial.polynomial.polyfit(
        scaled_k, chi_values[use], 3
    )
    raw_intercept = float(coefficients[0])
    physical_intercept = max(raw_intercept, 0.0)
    return {
        "H": physical_intercept / peak_value,
        "raw_H": raw_intercept / peak_value,
        "intercept": physical_intercept,
        "raw_intercept": raw_intercept,
        "peak_value": peak_value,
        "peak_k": peak_k,
        "peak_index": peak_index,
        "fit_fraction": float(fit_fraction),
        "fit_k_max": float(fit_k_max),
        "fit_bin_count": int(np.count_nonzero(use)),
    }


def bootstrap_H(
    spectra,
    k_values,
    fit_fraction: float,
    replicates: int,
    seed: int,
):
    """Bootstrap realizations and recompute averaging, peak, fit, and H."""

    rng = np.random.default_rng(seed)
    sample_count = spectra.shape[0]
    estimates = np.empty(replicates, dtype=np.float64)
    raw_estimates = np.empty(replicates, dtype=np.float64)
    batch_size = 50

    cursor = 0
    while cursor < replicates:
        current = min(batch_size, replicates - cursor)
        sample_indices = rng.integers(
            0, sample_count, size=(current, sample_count)
        )
        means = spectra[sample_indices].mean(axis=1)
        for offset, mean_spectrum in enumerate(means):
            estimate = cubic_intercept_H(
                k_values, mean_spectrum, fit_fraction
            )
            estimates[cursor + offset] = estimate["H"]
            raw_estimates[cursor + offset] = estimate["raw_H"]
        cursor += current

    return {
        "replicates": int(replicates),
        "ci_95": [
            float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)),
        ],
        "median": float(np.median(estimates)),
        "raw_ci_95": [
            float(np.percentile(raw_estimates, 2.5)),
            float(np.percentile(raw_estimates, 97.5)),
        ],
        "fraction_clipped_at_zero": float(np.mean(raw_estimates < 0.0)),
    }


def summarize(spectra, phase_fractions, k_values, conditions, bootstrap_replicates):
    """Compute ensemble H estimates, window sensitivity, and bootstrap CIs."""

    summary = {
        "H_threshold": H_THRESHOLD,
        "primary_fit_fraction_k_over_2pi_sqrt_rho": PRIMARY_FIT_FRACTION,
        "fit_sensitivity_fractions_k_over_2pi_sqrt_rho": list(FIT_FRACTIONS),
        "conditions": {},
    }
    for condition_index, condition in enumerate(conditions):
        condition_summary = {}
        for graph_index, graph_type in enumerate(GRAPH_TYPES):
            group_spectra = spectra[condition_index, graph_index]
            ensemble_spectrum = group_spectra.mean(axis=0)
            primary = cubic_intercept_H(
                k_values, ensemble_spectrum, PRIMARY_FIT_FRACTION
            )
            sensitivity = {}
            for fraction in FIT_FRACTIONS:
                estimate = cubic_intercept_H(
                    k_values, ensemble_spectrum, fraction
                )
                sensitivity[f"{fraction:.2f}"] = {
                    "H": float(estimate["H"]),
                    "raw_H": float(estimate["raw_H"]),
                    "fit_bin_count": int(estimate["fit_bin_count"]),
                }

            bootstrap = bootstrap_H(
                group_spectra,
                k_values,
                PRIMARY_FIT_FRACTION,
                bootstrap_replicates,
                BASE_SEED + 1000 * condition_index + graph_index,
            )
            condition_summary[graph_type] = {
                "H": float(primary["H"]),
                "raw_H": float(primary["raw_H"]),
                "chi_zero": float(primary["intercept"]),
                "raw_chi_zero": float(primary["raw_intercept"]),
                "chi_peak": float(primary["peak_value"]),
                "k_peak": float(primary["peak_k"]),
                "fit_bin_count": int(primary["fit_bin_count"]),
                "effectively_hyperuniform": bool(primary["H"] <= H_THRESHOLD),
                "bootstrap": bootstrap,
                "fit_window_sensitivity": sensitivity,
                "phase_fraction_mean": float(
                    phase_fractions[condition_index, graph_index].mean()
                ),
                "phase_fraction_std": float(
                    phase_fractions[condition_index, graph_index].std(
                        ddof=1
                        if phase_fractions.shape[2] > 1
                        else 0
                    )
                ),
            }
        summary["conditions"][condition] = condition_summary
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--npix", type=int, default=DEFAULT_NPIX)
    parser.add_argument(
        "--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--max-radial-mode", type=int, default=MAX_RADIAL_MODE
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help=(
            "Conditions to analyze; use 'poppi' or names of the form "
            "'huppi_a<nonnegative number>'."
        ),
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not retain the regenerated point and graph arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite only for an intentional replacement."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    raw_root = None if args.no_raw else output_dir / "raw"

    conditions = list(args.conditions)
    if len(set(conditions)) != len(conditions):
        raise ValueError("Condition names must be unique")
    for condition in conditions:
        condition_settings(condition)
    tasks = []
    for selected_index, condition in enumerate(conditions):
        canonical_index = condition_seed_index(condition)
        for ensemble_index in range(args.ensemble_size):
            seed = realization_seed(canonical_index, ensemble_index)
            tasks.append(
                (
                    condition,
                    selected_index,
                    ensemble_index,
                    seed,
                    args.npix,
                    args.max_radial_mode,
                    str(raw_root) if raw_root is not None else None,
                )
            )

    print(
        f"Paired grouped-H generation: {len(conditions)} conditions x "
        f"{args.ensemble_size} realizations x {len(GRAPH_TYPES)} networks; "
        f"npix={args.npix}; workers={args.workers}",
        flush=True,
    )
    started = time.time()
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_realization, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.time() - started
                eta = elapsed / completed * (len(tasks) - completed)
                print(
                    f"  {completed}/{len(tasks)} point patterns; "
                    f"elapsed={elapsed:.1f}s; ETA={eta:.1f}s; "
                    f"errors={len(errors)}",
                    flush=True,
                )

    if errors:
        with (output_dir / "errors.json").open("w") as handle:
            json.dump(errors, handle, indent=2)
        raise RuntimeError(
            f"{len(errors)} realizations failed; see {output_dir / 'errors.json'}"
        )

    results.sort(key=lambda result: (
        result["condition_index"], result["ensemble_index"]
    ))
    k_values = results[0]["k_values"]
    bin_count = len(k_values)
    spectra = np.empty(
        (
            len(conditions),
            len(GRAPH_TYPES),
            args.ensemble_size,
            bin_count,
        ),
        dtype=np.float64,
    )
    phase_fractions = np.empty(
        (len(conditions), len(GRAPH_TYPES), args.ensemble_size),
        dtype=np.float64,
    )
    seeds = np.empty(
        (len(conditions), args.ensemble_size), dtype=np.uint32
    )
    realization_seconds = np.empty(
        (len(conditions), args.ensemble_size), dtype=np.float64
    )

    for result in results:
        ci = result["condition_index"]
        ei = result["ensemble_index"]
        if not np.array_equal(result["k_values"], k_values):
            raise RuntimeError("Inconsistent radial k grid")
        spectra[ci, :, ei, :] = result["spectra"]
        phase_fractions[ci, :, ei] = result["phase_fractions"]
        seeds[ci, ei] = result["seed"]
        realization_seconds[ci, ei] = result["elapsed_seconds"]

    np.savez_compressed(
        output_dir / "spectra.npz",
        spectra=spectra,
        k_values=k_values,
        phase_fractions=phase_fractions,
        seeds=seeds,
        realization_seconds=realization_seconds,
        conditions=np.asarray(conditions),
        graph_types=np.asarray(GRAPH_TYPES),
    )

    print("Computing ensemble fits and bootstrap intervals...", flush=True)
    summary = summarize(
        spectra,
        phase_fractions,
        k_values,
        conditions,
        args.bootstrap_replicates,
    )
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    metadata = {
        "created_unix_time": time.time(),
        "elapsed_seconds": time.time() - started,
        "lattice_size": LATTICE_SIZE,
        "points_per_realization": LATTICE_SIZE**2,
        "box_length": BOX_LENGTH,
        "point_density": POINT_DENSITY,
        "beam_half_width": BEAM_HALF_WIDTH,
        "beam_full_width": 2.0 * BEAM_HALF_WIDTH,
        "npix": args.npix,
        "pixels_across_beam": 2.0 * BEAM_HALF_WIDTH / (BOX_LENGTH / args.npix),
        "ensemble_size": args.ensemble_size,
        "conditions": conditions,
        "condition_parameters": {
            condition: condition_settings(condition)
            for condition in conditions
        },
        "graph_types": list(GRAPH_TYPES),
        "bin_width_in_kmin": BIN_WIDTH_IN_KMIN,
        "max_radial_mode": args.max_radial_mode,
        "base_seed": BASE_SEED,
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_fit_fraction_k_over_2pi_sqrt_rho": PRIMARY_FIT_FRACTION,
        "fit_sensitivity_fractions_k_over_2pi_sqrt_rho": list(FIT_FRACTIONS),
        "raw_arrays_retained": not args.no_raw,
        "numpy_version": np.__version__,
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved new data under {output_dir}")
    print("\nH summary (primary cubic fit):")
    for condition in conditions:
        for graph_type in GRAPH_TYPES:
            record = summary["conditions"][condition][graph_type]
            lower, upper = record["bootstrap"]["ci_95"]
            print(
                f"  {condition:12s} {graph_type:22s} "
                f"H={record['H']:.4e} "
                f"95% CI=({lower:.4e}, {upper:.4e})"
            )


if __name__ == "__main__":
    main()

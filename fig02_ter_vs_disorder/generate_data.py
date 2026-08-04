#!/usr/bin/env python3
"""Generate raw data and network metrics for manuscript Figure 2.

For each distribution type (e.g. "classI" and "random"), it:
  - Generates point patterns (using a lattice plus perturbation for hyperuniform types,
    or random uniform points for "random").
  - Constructs network graphs using several graph construction methods.
  - Saves the raw data (point patterns and graph data) to disk.
  - Computes network metrics (e.g. Total Effective Resistance, TER) for each graph.
  - Saves the computed metrics and alpha values for later analysis.

All configuration parameters are read from config.py.
"""

import os
import sys
import numpy as np

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures
import config  # Use the shared configuration

# Get configuration parameters for data generation
cfg = config.get_data_params()
# Set up additional parameters for this experiment:
# For "classI", we scan a range of alpha values.
ALPHA_VALUES = np.linspace(0.1, 2.0, 20)
# For other distributions, we use a single alpha value from the config or a default.
def get_alpha_list(dist):
    if dist == "classI":
        return ALPHA_VALUES
    elif dist == "ordered":
        return [cfg["alpha_for_ordered"]]
    elif dist in ["classII", "classIII"]:
        return [cfg["alpha_for_hyperuniform"]]
    elif dist == "random":
        return [0.1]  # filename sentinel; alpha does not affect Poisson patterns
    else:
        return [cfg["alpha_for_hyperuniform"]]

# Use base directory from config (e.g. "data" as in config.py)
BASE_DIR = cfg["base_dir"]
POINTS_DIR = os.path.join(BASE_DIR, "points")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

# Create directories if needed
for d in [POINTS_DIR, GRAPHS_DIR, ANALYSIS_DIR]:
    os.makedirs(d, exist_ok=True)

# Define the graph types to be generated from the config
GRAPH_TYPES = config.get_plot_params()["graph_types"]

# Map our graph type names to the corresponding GraphGenerator methods.
def get_graph_constructors(Ggen):
    return {
        "delaunay": Ggen.periodic_delaunay_tessellation,
        "gabriel": Ggen.periodic_gabriel_graph,
        "delaunay_centroidal": Ggen.periodic_delaunay_centroidal,
        "voronoi_pruned": Ggen.periodic_voronoi_tessellation
    }

# Ensemble size from config
ENSEMBLE_SIZE = cfg["ensemble_size"]
# Lattice size and dimensions
N = cfg["lattice_size"]
DIM = cfg["dimensions"]
# Lattice spacing parameter (if needed)
C = cfg.get("C", 1.0)

# List of distribution types to process. Here we focus on "classI" and "random".
DIST_TYPES = ["classI", "random"]

print("Starting data generation...\n")
# Loop over distribution types and corresponding alpha values.
for dist in DIST_TYPES:
    alpha_list = get_alpha_list(dist)
    for alpha in alpha_list:
        print(f"Processing distribution '{dist}' with alpha={alpha:.2f}")
        for ens_idx in range(ENSEMBLE_SIZE):
            # Generate point pattern:
            if dist == "random":
                num_points = N * N
                points = np.random.uniform(0, N, size=(num_points, DIM))
            else:
                # For hyperuniform/ordered distributions, use the lattice generator.
                lat_gen = LatticeGenerator(size=N, dimensions=DIM, C=C)
                lat_gen.generate_lattice()
                lat_gen.perturb_lattice(alpha=alpha)
                points = lat_gen.get_points()

            # Save raw point pattern
            points_fname = f"{dist}_alpha{alpha:.2f}_ens{ens_idx}_points.npy"
            np.save(os.path.join(POINTS_DIR, points_fname), points)

            # Construct graphs from the points
            box_size = np.array([N, N])
            Ggen = GraphGenerator(points, box_size)
            constructors = get_graph_constructors(Ggen)

            for gtype in GRAPH_TYPES:
                # Generate graph (nodes and edge array)
                nodes, edges = constructors[gtype]()
                base_name = f"{dist}_alpha{alpha:.2f}_ens{ens_idx}_{gtype}"
                np.save(os.path.join(GRAPHS_DIR, base_name + "_nodes.npy"), nodes)
                np.save(os.path.join(GRAPHS_DIR, base_name + "_edges.npy"), edges)

            print(f"  Saved ensemble {ens_idx+1}/{ENSEMBLE_SIZE} for alpha={alpha:.2f}")
        print("")

print("Raw data generation complete.\n")

# -------------------------------
# Compute Network Metrics (TER)
# -------------------------------

# Define a normalization function (as in the original code)
def normalization_factor(N_val, edge_array):
    num_nodes = len(np.unique(edge_array[:, :2]))
    box_area = N_val * N_val
    return 1 / (num_nodes**2 * np.log(num_nodes))

NM = Network_Measures()

# We will compute metrics for each graph type for "classI" and "random" separately.
def compute_metrics(dist):
    metrics = {g: {"ter": [], "edge_sum": [], "node_count": [], "box_area": []} for g in GRAPH_TYPES}
    alpha_list = get_alpha_list(dist)
    for alpha in alpha_list:
        for gtype in GRAPH_TYPES:
            ensemble_vals = {"ter": [], "edge_sum": [], "node_count": [], "box_area": []}
            for ens_idx in range(ENSEMBLE_SIZE):
                edges_fname = os.path.join(
                    GRAPHS_DIR, f"{dist}_alpha{alpha:.2f}_ens{ens_idx}_{gtype}_edges.npy"
                )
                if not os.path.exists(edges_fname):
                    print(f"Warning: missing file {edges_fname}")
                    continue
                edge_array = np.load(edges_fname)
                ter = NM.compute_effective_resistance(edge_array)
                edge_sum = np.sum(edge_array[:, 2])
                node_count = len(np.unique(edge_array[:, :2]))
                norm_factor = normalization_factor(N, edge_array)
                ter_norm = ter * norm_factor

                ensemble_vals["ter"].append(ter_norm)
                ensemble_vals["edge_sum"].append(edge_sum)
                ensemble_vals["node_count"].append(node_count)
                ensemble_vals["box_area"].append(N * N)
            # Average over ensemble realizations (or set NaN if missing)
            for metric in ensemble_vals:
                if ensemble_vals[metric]:
                    mean_val = np.mean(ensemble_vals[metric])
                else:
                    mean_val = np.nan
                metrics[gtype][metric].append(mean_val)
    return alpha_list, metrics

print("Computing network metrics for 'classI' patterns...")
alpha_classI, metrics_classI = compute_metrics("classI")
print("Computing network metrics for 'random' patterns...")
_, metrics_random = compute_metrics("random")

# Save metrics and alpha values for later plotting
np.save(os.path.join(ANALYSIS_DIR, "alpha_values.npy"), alpha_classI)
np.save(os.path.join(ANALYSIS_DIR, "classI_network_metrics.npy"), metrics_classI)
np.save(os.path.join(ANALYSIS_DIR, "random_network_metrics.npy"), metrics_random)

print("\nNetwork metrics computed and saved in the analysis directory.")
print("Data generation complete!")

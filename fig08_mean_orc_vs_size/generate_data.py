#!/usr/bin/env python3

"""
generate_data.py

Generates data for manuscript Figure 8, analyzing curvature across lattice sizes
for hyperuniform classes and random patterns.
"""

import os
import sys
import numpy as np

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures
from config import get_data_generation_params

# Get parameters
params = get_data_generation_params()

BASE_DIR = params["base_dir"]
dimensions = params["dimensions"]
ensemble_size = params["ensemble_size"]
orc_alpha = params["orc_alpha"]
# Delaunay and Gabriel use the longest range; the other plot ranges are subsets.
lattice_size_range = params["lattice_size_ranges"]["delaunay"]
distribution_classes = params["distribution_classes"]
graph_types = params["graph_types"]

def create_output_dirs(base_dir):
    """Create required output directories."""
    os.makedirs(f"{base_dir}/points", exist_ok=True)
    os.makedirs(f"{base_dir}/graphs", exist_ok=True)
    os.makedirs(f"{base_dir}/curvature", exist_ok=True)

def generate_hyperuniform_points():
    """
    Generate hyperuniform point patterns for all classes and lattice sizes,
    with fixed alpha=1 for perturbations.
    """
    for lsize in lattice_size_range:
        for dist_type in distribution_classes:
            for ens_idx in range(ensemble_size):
                lat_gen = LatticeGenerator(size=lsize, dimensions=dimensions, C=1.0)
                lat_gen.generate_lattice()
                if dist_type == "classI":
                    lat_gen.perturb_lattice(alpha=1)
                elif dist_type == "classII":
                    lat_gen.perturb_lattice_cauchy(alpha=1)
                elif dist_type == "classIII":
                    lat_gen.perturb_lattice_pareto(alpha=1)
                points = lat_gen.get_points()
                fname_points = f"{dist_type}_lsize{lsize}_ens{ens_idx}_points.npy"
                np.save(f"{BASE_DIR}/points/{fname_points}", points)

def generate_random_points():
    """
    Generate random point patterns such that for every hyperuniform pattern
    there is one corresponding random pattern. This loops only over lattice
    size and ensemble, ensuring that the ensemble size exactly matches
    that of the hyperuniform patterns.
    """
    for lsize in lattice_size_range:
        for ens_idx in range(ensemble_size):
            lat_gen = LatticeGenerator(size=lsize, dimensions=dimensions, C=1.0)
            points = lat_gen.generate_random()
            fname_points = f"random_pattern_lsize{lsize}_ens{ens_idx}_points.npy"
            np.save(f"{BASE_DIR}/points/{fname_points}", points)

def construct_graphs(is_random=False):
    """
    Construct all graph types for a given point pattern.
    If is_random=True, iterate over random patterns (one per lattice size and ensemble).
    Otherwise, iterate over hyperuniform patterns.

    Note: _process_single_graph internally generates graphs for all types.
    """
    if is_random:
        for lsize in lattice_size_range:
            for ens_idx in range(ensemble_size):
                fname_points = f"random_pattern_lsize{lsize}_ens{ens_idx}_points.npy"
                _process_single_graph(fname_points)
    else:
        for lsize in lattice_size_range:
            for dist_type in distribution_classes:
                for ens_idx in range(ensemble_size):
                    fname_points = f"{dist_type}_lsize{lsize}_ens{ens_idx}_points.npy"
                    _process_single_graph(fname_points)

def _process_single_graph(fname_points):
    """Helper function to process a single point set and create all graph types."""
    fpath_points = f"{BASE_DIR}/points/{fname_points}"
    if not os.path.exists(fpath_points):
        return

    points = np.load(fpath_points)
    # Box size is derived from the square root of the number of points.
    L = (len(points))**0.5
    box_size = np.array([L, L])
    Ggen = GraphGenerator(points, box_size)

    graph_constructors = {
        "delaunay": Ggen.periodic_delaunay_tessellation,
        "gabriel": Ggen.periodic_gabriel_graph,
        "delaunay_centroidal": Ggen.periodic_delaunay_centroidal,
        "voronoi_pruned": Ggen.periodic_voronoi_tessellation
    }

    for gtype, constructor in graph_constructors.items():
        nodes, edge_array = constructor()
        base_name = fname_points.replace("_points.npy", f"_{gtype}")
        np.save(f"{BASE_DIR}/graphs/{base_name}_nodes.npy", nodes)
        np.save(f"{BASE_DIR}/graphs/{base_name}_edges.npy", edge_array)

def compute_curvatures(is_random=False):
    """
    Compute Ollivier-Ricci curvature for all graphs.
    If is_random=True, iterate over random patterns (one per lattice size and ensemble).
    Otherwise, iterate over hyperuniform patterns.
    """
    NM = Network_Measures()
    if is_random:
        for gtype in graph_types:
            for lsize in lattice_size_range:
                for ens_idx in range(ensemble_size):
                    fname = f"random_pattern_lsize{lsize}_ens{ens_idx}_{gtype}"
                    _process_single_curvature(fname, NM)
    else:
        for gtype in graph_types:
            for lsize in lattice_size_range:
                for dist_type in distribution_classes:
                    for ens_idx in range(ensemble_size):
                        fname = f"{dist_type}_lsize{lsize}_ens{ens_idx}_{gtype}"
                        _process_single_curvature(fname, NM)

def _process_single_curvature(fname_base, NM):
    """Helper function to compute curvature for a single graph."""
    fpath_edges = f"{BASE_DIR}/graphs/{fname_base}_edges.npy"
    if not os.path.exists(fpath_edges):
        return
    edge_list = np.load(fpath_edges)
    or_result = NM.ollivier_ricci_curvature(edge_list, alpha=orc_alpha)
    np.save(f"{BASE_DIR}/curvature/{fname_base}_ollivier.npy", or_result)

if __name__ == "__main__":
    create_output_dirs(BASE_DIR)

    print("Generating hyperuniform point patterns...")
    generate_hyperuniform_points()

    print("Generating random point patterns...")
    generate_random_points()

    print("Constructing graphs for hyperuniform patterns...")
    construct_graphs(is_random=False)

    print("Constructing graphs for random patterns...")
    construct_graphs(is_random=True)

    print("Computing curvatures for hyperuniform patterns...")
    compute_curvatures(is_random=False)

    print("Computing curvatures for random patterns...")
    compute_curvatures(is_random=True)

    print("All done.")

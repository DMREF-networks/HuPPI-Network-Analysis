#!/usr/bin/env python3

"""
make_data_2_combined.py

Generates the data for manuscript Figure 7:
- Sweeps disorder parameter alpha for HuPPI point patterns
- Computes curvature for all graph types (Delaunay, Gabriel, Delaunay-centroidal, Voronoi)
- Generates one 50-realization PoPPI ensemble for comparison

Data is saved in subdirectories of BASE_DIR: points/, graphs/, curvature/.
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures
from config import get_data_generation_params

###############################################################################
#                       EXPERIMENT PARAMETERS
###############################################################################

# Get parameters
params = get_data_generation_params()

# Extract parameters
BASE_DIR = params["base_dir"]
dimensions = params["dimensions"]
ensemble_size = params["ensemble_size"]
orc_alpha = params["orc_alpha"]
alpha_values = params["alpha_values"]
distribution_classes = params["distribution_classes"]
graph_types = params["graph_types"]
lattice_size = params["lattice_size"]


def create_output_dirs(base_dir):
    """Create required output directories"""
    os.makedirs(f"{base_dir}/points", exist_ok=True)
    os.makedirs(f"{base_dir}/graphs", exist_ok=True)
    os.makedirs(f"{base_dir}/curvature", exist_ok=True)

def generate_hyperuniform_points():
    """Generate hyperuniform point patterns for all classes and alpha values"""
    for dist_type in distribution_classes:
        for alpha in alpha_values:
            for ens_idx in range(ensemble_size):
                lat_gen = LatticeGenerator(size=lattice_size, dimensions=dimensions, C=1.0)
                lat_gen.generate_lattice()

                if dist_type == "classI":
                    lat_gen.perturb_lattice(alpha=alpha)
                elif dist_type == "classII":
                    lat_gen.perturb_lattice_cauchy(alpha=alpha)
                elif dist_type == "classIII":
                    lat_gen.perturb_lattice_pareto(alpha=alpha)

                points = lat_gen.get_points()
                fname_points = f"{dist_type}_alpha{alpha:.2f}_ens{ens_idx}_points.npy"
                np.save(f"{BASE_DIR}/points/{fname_points}", points)

def generate_random_points():
    """Generate the PoPPI ensemble stated in the manuscript."""
    for ens_idx in range(ensemble_size):
        total_points = lattice_size**dimensions
        lat_gen = LatticeGenerator(
            size=int(total_points**(1 / dimensions)),
            dimensions=dimensions,
            C=1.0,
        )
        points = lat_gen.generate_random()
        fname_points = f"random_ens{ens_idx}_points.npy"
        np.save(f"{BASE_DIR}/points/{fname_points}", points)

def construct_graphs(point_file_prefix, is_random=False):
    """Construct all graph types for given point pattern"""
    if is_random:
        for ens_idx in range(ensemble_size):
            fname_points = f"random_ens{ens_idx}_points.npy"
            _process_single_graph(fname_points)
        return

    for alpha in alpha_values:
        for ens_idx in range(ensemble_size):
            for dist_type in distribution_classes:
                fname_points = f"{dist_type}_alpha{alpha:.2f}_ens{ens_idx}_points.npy"
                _process_single_graph(fname_points)

def _process_single_graph(fname_points):
    """Helper function to process a single point set and create all graph types"""
    fpath_points = f"{BASE_DIR}/points/{fname_points}"
    if not os.path.exists(fpath_points):
        return

    points = np.load(fpath_points)
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

def compute_curvatures(file_prefix, is_random=False):
    """Compute Ollivier-Ricci curvature for all graphs"""
    NM = Network_Measures()
    if is_random:
        for ens_idx in range(ensemble_size):
            for gtype in graph_types:
                fname = f"random_ens{ens_idx}_{gtype}"
                _process_single_curvature(fname, NM)
        return

    for alpha in alpha_values:
        for ens_idx in range(ensemble_size):
            for dist_type in distribution_classes:
                for gtype in graph_types:
                    fname = f"{dist_type}_alpha{alpha:.2f}_ens{ens_idx}_{gtype}"
                    _process_single_curvature(fname, NM)

def _process_single_curvature(fname_base, NM):
    """Helper function to compute curvature for a single graph"""
    fpath_edges = f"{BASE_DIR}/graphs/{fname_base}_edges.npy"
    if not os.path.exists(fpath_edges):
        return

    edge_list = np.load(fpath_edges)
    or_result = NM.ollivier_ricci_curvature(edge_list, alpha=orc_alpha)
    np.save(f"{BASE_DIR}/curvature/{fname_base}_ollivier.npy", or_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population",
        choices=("all", "huppi", "poppi"),
        default="all",
        help="Regenerate both populations or only one population.",
    )
    args = parser.parse_args()

    create_output_dirs(BASE_DIR)

    if args.population in ("all", "huppi"):
        print("Generating HuPPI point patterns...")
        generate_hyperuniform_points()
        print("Constructing graphs for HuPPI patterns...")
        construct_graphs("", is_random=False)
        print("Computing curvatures for HuPPI patterns...")
        compute_curvatures("", is_random=False)

    if args.population in ("all", "poppi"):
        print("Generating PoPPI point patterns...")
        generate_random_points()
        print("Constructing graphs for PoPPI patterns...")
        construct_graphs("random_", is_random=True)
        print("Computing curvatures for PoPPI patterns...")
        compute_curvatures("random_", is_random=True)

    print("All done.")

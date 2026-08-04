#!/usr/bin/env python3

"""Generate the data used in manuscript Figure 6.

The active configuration compares two-dimensional HuPPI (internally
``classI``) and PoPPI (internally ``random``) point patterns for the Gabriel,
Delaunay, Delaunay-centroidal, and Voronoi constructions. The generator also
retains support for the other distribution classes recognized by the shared
network utilities.

Data are saved below ``BASE_DIR`` in ``points/``, ``graphs/``, and
``curvature/``.
"""

import os
import sys
import numpy as np

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import classes and functions from your library
from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures
from config import get_data_params

###############################################################################
#                     USER-DEFINED BASE DIRECTORY
###############################################################################
# Get parameters
params = get_data_params()

# Then replace the hardcoded values with params.get() calls
BASE_DIR = params["base_dir"]
distribution_classes = params["distribution_classes"]

###############################################################################
#                       EXPERIMENT PARAMETERS
###############################################################################
# Store alpha=0.0 for ordered patterns and alpha=1.0 for HuPPI patterns. The
# Poisson data use a sentinel because alpha does not parameterize that process.
alpha_for_ordered = params["alpha_for_ordered"]
alpha_for_hyperuniform = params["alpha_for_hyperuniform"]

dist_to_alpha_map = {
    "ordered": alpha_for_ordered,
    "classI": alpha_for_hyperuniform,
    "classII": alpha_for_hyperuniform,
    "classIII": alpha_for_hyperuniform,
    "random": 999.0,
}

# Graph types to generate
graph_types = params["graph_types"]

ensemble_size = params["ensemble_size"]
lattice_size = params["lattice_size"]
dimensions = params["dimensions"]

# Ollivier-Ricci parameter
orc_alpha = params["orc_alpha"]

###############################################################################
#                    CREATE OUTPUT DIRECTORIES
###############################################################################
def create_output_dirs(base_dir):
    """
    Create the subdirectories 'points', 'graphs', and 'curvature'
    within the given base directory.
    """
    os.makedirs(os.path.join(base_dir, "points"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "curvature"), exist_ok=True)

###############################################################################
#                STEP 1: Generate Point Sets for Each Distribution
###############################################################################
def generate_point_sets():
    """
    For each distribution type in 'distribution_classes' and each ensemble index,
    generate a point set (lattice plus perturbations, or a Poisson pattern),
    then save the resulting points as a .npy file in BASE_DIR/points/.
    """
    for dist_type in distribution_classes:
        alpha_val = dist_to_alpha_map[dist_type]
        for ens_idx in range(ensemble_size):
            lat_gen = LatticeGenerator(size=lattice_size, dimensions=dimensions, C=1.0)

            if dist_type == "ordered":
                # alpha=0 => no perturbation
                lat_gen.generate_lattice()
                points = lat_gen.get_points()

            elif dist_type == "classI":
                lat_gen.generate_lattice()
                lat_gen.perturb_lattice(alpha=alpha_for_hyperuniform)
                points = lat_gen.get_points()

            elif dist_type == "classII":
                lat_gen.generate_lattice()
                lat_gen.perturb_lattice_cauchy(alpha=alpha_for_hyperuniform)
                points = lat_gen.get_points()

            elif dist_type == "classIII":
                lat_gen.generate_lattice()
                lat_gen.perturb_lattice_pareto(alpha=alpha_for_hyperuniform)
                points = lat_gen.get_points()

            elif dist_type == "random":
                # For random points, create a new generator with the same total number of points
                total_points = lattice_size**dimensions
                lat_gen = LatticeGenerator(size=int(total_points**(1/dimensions)), dimensions=dimensions, C=1.0)
                points = lat_gen.generate_random()

            # Save points
            fname_points = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_points.npy"
            save_path = os.path.join(BASE_DIR, "points", fname_points)
            np.save(save_path, points)

###############################################################################
#          STEP 2: Construct Graphs (Including the Two New Types)
###############################################################################
def construct_graphs():
    """
    For each distribution type, ensemble, and graph type,
    load the corresponding point set (if it exists),
    build the graph (edgelist) using the specified method (gabriel, delaunay, etc.),
    and save the edgelist as a .npy file in BASE_DIR/graphs/.
    """
    for dist_type in distribution_classes:
        alpha_val = dist_to_alpha_map[dist_type]
        for ens_idx in range(ensemble_size):
            # Load points
            fname_points = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_points.npy"
            fpath_points = os.path.join(BASE_DIR, "points", fname_points)
            if not os.path.exists(fpath_points):
                continue

            points = np.load(fpath_points)
            L = len(points[:,0])**0.5
            box_size = np.array([L,L])
            Ggen = GraphGenerator(points, box_size)

            # Build each graph type
            for gtype in graph_types:
                if gtype == "gabriel":
                    nodes, edge_array = Ggen.periodic_gabriel_graph()
                elif gtype == "delaunay":
                    nodes, edge_array = Ggen.periodic_delaunay_tessellation()
                elif gtype == "delaunay_centroidal":
                    nodes, edge_array = Ggen.periodic_delaunay_centroidal()
                elif gtype == "voronoi_pruned":
                    nodes, edge_array = Ggen.periodic_voronoi_tessellation()

                # Save both nodes and edges
                fname_nodes = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_{gtype}_nodes.npy"
                fname_edges = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_{gtype}_edges.npy"

                np.save(os.path.join(BASE_DIR, "graphs", fname_nodes), nodes)
                np.save(os.path.join(BASE_DIR, "graphs", fname_edges), edge_array)

###############################################################################
#                 STEP 3: Compute Ollivier-Ricci Curvature
###############################################################################
def compute_curvatures():
    """
    For each distribution type, ensemble, and graph type,
    load the edgelist (if it exists),
    compute Ollivier-Ricci curvature, and save the results
    as a .npy file in BASE_DIR/curvature/.
    """
    NM = Network_Measures()
    for dist_type in distribution_classes:
        alpha_val = dist_to_alpha_map[dist_type]
        for ens_idx in range(ensemble_size):
            for gtype in graph_types:
                fname_edges = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_{gtype}_edges.npy"
                fpath_edges = os.path.join(BASE_DIR, "graphs", fname_edges)
                if not os.path.exists(fpath_edges):
                    continue

                edge_list = np.load(fpath_edges)
                # edge_list => shape (M, 3): [node1, node2, length]

                # The second 'alpha' below is the Ollivier-Ricci alpha (orc_alpha),
                # not to be confused with the distribution alpha.
                or_result = NM.ollivier_ricci_curvature(edge_list, alpha=orc_alpha)
                # or_result => shape (M, 3): [node1, node2, ricci_curvature]

                # Save curvature
                fname_curv = f"{dist_type}_alpha{alpha_val:.1f}_ens{ens_idx}_{gtype}_ollivier.npy"
                save_path = os.path.join(BASE_DIR, "curvature", fname_curv)
                np.save(save_path, or_result)

###############################################################################
#                               MAIN
###############################################################################
if __name__ == "__main__":
    # Create necessary subdirectories in BASE_DIR
    create_output_dirs(BASE_DIR)

    print(f"Generating 2D point sets into '{BASE_DIR}/points'...")
    generate_point_sets()

    print(f"Constructing graphs (Gabriel, Delaunay, Delaunay-centroidal, Voronoi-pruned) into '{BASE_DIR}/graphs'...")
    construct_graphs()

    print(f"Computing Ollivier-Ricci curvatures, saving into '{BASE_DIR}/curvature'...")
    compute_curvatures()

    print("\nDone! Data is saved in:")
    print(f"  {BASE_DIR}/points")
    print(f"  {BASE_DIR}/graphs")
    print(f"  {BASE_DIR}/curvature")
    print("You may now use another script to plot your figures based on this data.")

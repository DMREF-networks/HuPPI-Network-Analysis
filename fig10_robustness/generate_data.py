#!/usr/bin/env python3
"""
generate_robustness_data.py

This script generates network data for each combination of network class (Class I and Random)
and graph type (delaunay, gabriel, delaunay_centroidal, voronoi). For each realization, it
simulates edge removal using two strategies:
  1. Lowest-curvature-first removal (ascending order)
  2. Random removal

The resulting removal curves (largest connected component vs. fraction of edges removed)
are stored (for each ensemble) in a nested dictionary and saved to disk.
"""

import os
import sys
import numpy as np
import networkx as nx
import random

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import network generation and measures modules
from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures

# ---------------- Global Parameters ----------------
GRAPH_TYPES = ["delaunay", "gabriel", "delaunay_centroidal", "voronoi"]
GRAPH_FUNCTION_MAPPING = {
    "delaunay": "periodic_delaunay_tessellation",
    "gabriel": "periodic_gabriel_graph",
    "delaunay_centroidal": "periodic_delaunay_centroidal",
    "voronoi": "periodic_voronoi_tessellation"
}

# Network generation parameters
N = 15         # Lattice size for Class I networks
dims = 2
C = 1.0
ensemble_classI = 20   # Number of realizations for Class I networks
ensemble_random = 20   # Number of realizations for Random networks
alpha = 1.0     # Perturbation strength for Class I (matches paper a=1)

# Removal simulation parameters
phi_values = np.linspace(0, 1, 21)  # 21 steps from 0 to 1

# ---------------- Helper Functions ----------------
def create_graph_from_data(nodes, edges):
    """
    Create an unweighted NetworkX graph from the given nodes and edges.
    Nodes: numpy array (positions are not used; only count matters)
    Edges: numpy array with columns [i, j, length]
    """
    G = nx.Graph()
    num_nodes = nodes.shape[0]
    G.add_nodes_from(range(num_nodes))
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        G.add_edge(i, j)
    return G

def simulate_removal(G, removal_order, phi_values):
    """
    Simulate removal of edges from graph G in the specified removal_order.
    At each target fraction phi of the total edges removed, compute the
    size of the largest connected component (LCC) as a fraction of the original node count.

    Returns:
      (phis, lcc_fractions): Tuple of arrays.
    """
    G_sim = G.copy()
    total_edges = len(removal_order)
    n0 = G_sim.number_of_nodes()
    lcc_results = {}
    current_removed = 0
    for phi in phi_values:
        target = int(np.floor(phi * total_edges))
        while current_removed < target and current_removed < total_edges:
            edge = removal_order[current_removed]
            if G_sim.has_edge(edge[0], edge[1]):
                G_sim.remove_edge(edge[0], edge[1])
            current_removed += 1
        if G_sim.number_of_edges() > 0:
            largest_cc = max(nx.connected_components(G_sim), key=len)
            lcc_size = len(largest_cc)
        else:
            lcc_size = 0
        lcc_results[phi] = lcc_size / n0
    phis = np.array(sorted(lcc_results.keys()))
    lcc_fractions = np.array([lcc_results[phi] for phi in phis])
    return phis, lcc_fractions

def simulate_robustness(rec):
    """
    For a given network realization (rec) containing:
      - "nodes": node positions,
      - "edges": edge array [i, j, length],
      - "curvature": array with columns [i, j, curvature],
    simulate the edge removal process using two strategies:
      (a) Lowest-curvature-first removal (ascending order)
      (b) Random removal.

    Returns a dictionary with keys "lowest" and "random", each mapping to (phis, lcc_fractions).
    """
    G = create_graph_from_data(rec["nodes"], rec["edges"])
    # Create a list of edges with curvature values
    edges_list = [(int(c[0]), int(c[1]), c[2]) for c in rec["curvature"]]
    lowest_order = sorted(edges_list, key=lambda x: x[2])
    random_order = edges_list.copy()
    random.shuffle(random_order)
    phis_lowest, lcc_lowest = simulate_removal(G, lowest_order, phi_values)
    # For random removal, use a fresh copy of the graph
    G2 = create_graph_from_data(rec["nodes"], rec["edges"])
    phis_random, lcc_random = simulate_removal(G2, random_order, phi_values)
    return {"lowest": (phis_lowest, lcc_lowest), "random": (phis_random, lcc_random)}

def generate_network_data():
    """
    Generate network data for each combination of network class and graph type.
    For Class I: use perturbed lattices; for Random: use uniformly random points.

    Returns a nested dictionary structured as:
      data = {
         "classI": { "delaunay": [rec, ...], "gabriel": [rec, ...], ... },
         "random": { "delaunay": [rec, ...], "gabriel": [rec, ...], ... }
      }
    Each realization (rec) contains "points", "nodes", "edges", "curvature", and (for Class I) "alpha".
    """
    NM = Network_Measures()
    data = {"classI": {g: [] for g in GRAPH_TYPES},
            "random": {g: [] for g in GRAPH_TYPES}}

    raw_points_dir = os.path.join(os.path.dirname(__file__), "data", "raw", "points")
    raw_graphs_dir = os.path.join(os.path.dirname(__file__), "data", "raw", "graphs")
    os.makedirs(raw_points_dir, exist_ok=True)
    os.makedirs(raw_graphs_dir, exist_ok=True)

    # Class I networks: perturbed lattices
    for ens in range(ensemble_classI):
        lat_gen = LatticeGenerator(size=N, dimensions=dims, C=C)
        lat_gen.generate_lattice()
        lat_gen.perturb_lattice(alpha=alpha)
        points = lat_gen.get_points()
        box_size = np.array([N, N])
        Ggen = GraphGenerator(points, box_size)
        np.save(os.path.join(raw_points_dir,
                             f"classI_alpha{alpha:.1f}_ens{ens}_points.npy"), points)
        for gtype in GRAPH_TYPES:
            method = getattr(Ggen, GRAPH_FUNCTION_MAPPING[gtype])
            nodes, edge_array = method()
            curvature_array = NM.ollivier_ricci_curvature(edge_array, alpha=0)
            base = f"classI_alpha{alpha:.1f}_ens{ens}_{gtype}"
            np.save(os.path.join(raw_graphs_dir, base + "_nodes.npy"), nodes)
            np.save(os.path.join(raw_graphs_dir, base + "_edges.npy"), edge_array)
            rec = {"points": points, "nodes": nodes, "edges": edge_array,
                   "curvature": curvature_array, "alpha": alpha}
            data["classI"][gtype].append(rec)
            print(f"Class I: {gtype} ensemble {ens} generated.")

    # Random networks
    for ens in range(ensemble_random):
        num_points = N * N
        points = np.random.uniform(0, N, size=(num_points, dims))
        box_size = np.array([N, N])
        Ggen = GraphGenerator(points, box_size)
        np.save(os.path.join(raw_points_dir,
                             f"random_ens{ens}_points.npy"), points)
        for gtype in GRAPH_TYPES:
            method = getattr(Ggen, GRAPH_FUNCTION_MAPPING[gtype])
            nodes, edge_array = method()
            curvature_array = NM.ollivier_ricci_curvature(edge_array, alpha=0)
            base = f"random_ens{ens}_{gtype}"
            np.save(os.path.join(raw_graphs_dir, base + "_nodes.npy"), nodes)
            np.save(os.path.join(raw_graphs_dir, base + "_edges.npy"), edge_array)
            rec = {"points": points, "nodes": nodes, "edges": edge_array,
                   "curvature": curvature_array}
            data["random"][gtype].append(rec)
            print(f"Random: {gtype} ensemble {ens} generated.")

    return data

def main():
    # Generate network data
    data = generate_network_data()

    # Prepare structure for simulation results:
    # results[removal_strategy][graph_type][net_class] will hold a list of (phis, lcc) tuples.
    results = {"lowest": {g: {"classI": [], "random": []} for g in GRAPH_TYPES},
               "random": {g: {"classI": [], "random": []} for g in GRAPH_TYPES}}

    # Loop over network classes and graph types, and simulate edge removals
    for net_class in ["classI", "random"]:
        for gtype in GRAPH_TYPES:
            for rec in data[net_class][gtype]:
                sim = simulate_robustness(rec)
                results["lowest"][gtype][net_class].append(sim["lowest"])
                results["random"][gtype][net_class].append(sim["random"])
            print(f"Simulated removal for {net_class} - {gtype}.")

    # Save the results to a file
    output_file = "robustness_data.npy"
    np.save(output_file, results)
    print(f"Robustness simulation data saved to {output_file}")

if __name__ == "__main__":
    main()

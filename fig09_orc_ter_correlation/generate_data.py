#!/usr/bin/env python3
"""
Generate network data for the manuscript Figure 9 edge-correlation analysis.

For each HuPPI and PoPPI network class (stored under the internal keys
``classI`` and ``random``) and each graph type, this script generates 10
independent realizations and computes:
    - The point set and nodes,
    - The edge array (with columns: [i, j, edge length]),
    - The edge-level Ollivier–Ricci curvature (columns: [i, j, curvature]),
    - The TER edge contribution: ΔR_tot(e) = R_tot(G \ e) − R_tot(G), each
      normalized by N log(N).

Realizations are processed in parallel via ProcessPoolExecutor; set
OMP_NUM_THREADS=1 for single-threaded BLAS to avoid worker oversubscription.

Output: data/plot7_data/plot7_data.pkl
"""

import os
import sys
import pickle
import numpy as np
import scipy.sparse as sp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network.generators import LatticeGenerator, GraphGenerator
from utils.network.measures import Network_Measures

# ── Parameters ───────────────────────────────────────────────────────

GRAPH_TYPES = ["delaunay", "gabriel", "delaunay_centroidal", "voronoi"]
GRAPH_FUNCTION_MAPPING = {
    "delaunay":            "periodic_delaunay_tessellation",
    "gabriel":             "periodic_gabriel_graph",
    "delaunay_centroidal": "periodic_delaunay_centroidal",
    "voronoi":             "periodic_voronoi_tessellation",
}

N            = 33    # Lattice size (33×33 = 1089 points)
DIMS         = 2
C            = 1.0
ALPHA        = 1.0
ENSEMBLE_CLASSI = 10
ENSEMBLE_RANDOM = 10

# Worker count for the realization-level parallelism. Each worker also runs
# its own ORC + TER pipeline, so single-threaded BLAS keeps the load sane.
N_WORKERS = min(20, os.cpu_count() or 1)

RAW_POINTS_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "points")
RAW_GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "graphs")


# ── Edge-level computations ──────────────────────────────────────────

def compute_edge_resistance_contributions(nodes, edge_array):
    """
    For each edge e = (i,j) of the network with conductance c_e = 1/length(e),
    compute the change in total effective resistance when that edge is removed:

        ΔR_tot(e) = R_tot(G \\ e) − R_tot(G),

    normalized by N log(N).

    Uses a single Laplacian eigendecomposition plus a rank-one pseudoinverse
    update per edge, giving O(N^3 + M N) instead of the O(M N^3) brute-force
    approach that recomputes R_tot from scratch for every removed edge.

    Identity used (for non-bridge edges, where the kernel of the Laplacian is
    unchanged by removal):
        Δtr(L⁺) = c_e ||L⁺ b_e||² / (1 − c_e b_eᵀ L⁺ b_e),
    where b_e = e_i − e_j is the signed indicator of edge e and
    R_e := b_eᵀ L⁺ b_e is the dyadic effective resistance between i and j.

    For bridge edges (where 1 − c_e R_e ≤ tol), the removed graph is
    disconnected, so we set ΔR_tot(e) = NaN.
    """
    Nnodes = len(nodes)
    log_norm = np.log(Nnodes)

    node1 = edge_array[:, 0].astype(int)
    node2 = edge_array[:, 1].astype(int)
    weights = 1.0 / edge_array[:, 2]
    M = len(edge_array)

    # Build symmetric weighted adjacency, then Laplacian
    adj = sp.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([node1, node2]),
          np.concatenate([node2, node1]))),
        shape=(Nnodes, Nnodes),
    ).tocsr()
    degrees = np.array(adj.sum(axis=1)).flatten()
    L_dense = (sp.diags(degrees) - adj).toarray()

    # Pseudoinverse via eigendecomposition (single O(N^3) call)
    eigvals, eigvecs = np.linalg.eigh(L_dense)
    inv = np.zeros_like(eigvals)
    nonzero = eigvals > 1e-10
    inv[nonzero] = 1.0 / eigvals[nonzero]
    L_pinv = (eigvecs * inv) @ eigvecs.T

    diag = np.diag(L_pinv)
    R_e = diag[node1] + diag[node2] - 2.0 * L_pinv[node1, node2]
    # Columns of L_pinv at i, j; v = L_pinv @ b_e = (L_pinv[:, i] - L_pinv[:, j])
    v = L_pinv[:, node1] - L_pinv[:, node2]                # shape (Nnodes, M)
    v_norm_sq = np.einsum("ne,ne->e", v, v)

    denom = 1.0 - weights * R_e
    bridge = denom <= 1e-10
    delta_trace = np.where(bridge,
                           np.nan,
                           weights * v_norm_sq / np.where(bridge, 1.0, denom))

    # ΔR_tot = N × Δtr(L⁺); normalized contribution = ΔR_tot / (N log N) = Δtr / log N
    return delta_trace / log_norm


# ── Per-realization worker ───────────────────────────────────────────

def run_one_realization(args):
    """
    Generate one (data_key, gtype, ens) realization and return a record.
    args: (data_key, gtype, ens_idx, seed)
    """
    data_key, gtype, ens_idx, seed = args
    np.random.seed(seed)

    NM = Network_Measures()

    if data_key == "classI":
        lat_gen = LatticeGenerator(size=N, dimensions=DIMS, C=C)
        lat_gen.generate_lattice()
        lat_gen.perturb_lattice(alpha=ALPHA)
        points = lat_gen.get_points()
    else:  # "random" → Poisson
        num_points = N * N
        points = np.random.uniform(0, N, size=(num_points, DIMS))

    box_size = np.array([N, N])
    Ggen = GraphGenerator(points, box_size)
    method = getattr(Ggen, GRAPH_FUNCTION_MAPPING[gtype])
    nodes, edge_array = method()

    curvature_array = NM.ollivier_ricci_curvature(edge_array, alpha=0)
    eff_res = compute_edge_resistance_contributions(nodes, edge_array)

    # Save per-realization individual files for ConfigLib export
    label = "classI" if data_key == "classI" else "random"
    pts_fname = f"{label}_alpha{ALPHA:.1f}_ens{ens_idx}_points.npy"
    np.save(os.path.join(RAW_POINTS_DIR, pts_fname), points)
    base = f"{label}_alpha{ALPHA:.1f}_ens{ens_idx}_{gtype}"
    np.save(os.path.join(RAW_GRAPHS_DIR, base + "_nodes.npy"), nodes)
    np.save(os.path.join(RAW_GRAPHS_DIR, base + "_edges.npy"), edge_array)

    rec = {
        "points":               points,
        "nodes":                nodes,
        "edges":                edge_array,
        "curvature":            curvature_array,
        "effective_resistance": eff_res,
    }
    if data_key == "classI":
        rec["alpha"] = ALPHA
    return data_key, gtype, ens_idx, rec


# ── Driver ───────────────────────────────────────────────────────────

def generate_and_save_data(base_dir="data/plot7_data"):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(RAW_POINTS_DIR, exist_ok=True)
    os.makedirs(RAW_GRAPHS_DIR, exist_ok=True)

    data = {"classI": {g: [None] * ENSEMBLE_CLASSI for g in GRAPH_TYPES},
            "random": {g: [None] * ENSEMBLE_RANDOM for g in GRAPH_TYPES}}

    seeds = np.random.SeedSequence(20260427).generate_state(
        len(GRAPH_TYPES) * (ENSEMBLE_CLASSI + ENSEMBLE_RANDOM)
    )

    tasks = []
    si = 0
    for gtype in GRAPH_TYPES:
        for ens in range(ENSEMBLE_CLASSI):
            tasks.append(("classI", gtype, ens, int(seeds[si]))); si += 1
        for ens in range(ENSEMBLE_RANDOM):
            tasks.append(("random", gtype, ens, int(seeds[si]))); si += 1

    print(f"Fig 7 generation: {len(tasks)} tasks "
          f"(classI={ENSEMBLE_CLASSI}, random={ENSEMBLE_RANDOM} per gtype, "
          f"N={N}, alpha={ALPHA}), {N_WORKERS} workers")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one_realization, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futures), 1):
            data_key, gtype, ens_idx, rec = fut.result()
            data[data_key][gtype][ens_idx] = rec
            elapsed = time.time() - t0
            eta = elapsed / k * (len(tasks) - k)
            print(f"  {data_key:6s} {gtype:22s} ens {ens_idx:2d}  "
                  f"[{k}/{len(tasks)}]  elapsed {elapsed:.1f}s  ETA {eta:.1f}s",
                  flush=True)

    # Drop any None slots (shouldn't happen if all succeed)
    for k1 in data:
        for g in data[k1]:
            data[k1][g] = [r for r in data[k1][g] if r is not None]

    save_path = os.path.join(base_dir, "plot7_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nData saved to {save_path}")
    return data


def main():
    generate_and_save_data()


if __name__ == "__main__":
    main()

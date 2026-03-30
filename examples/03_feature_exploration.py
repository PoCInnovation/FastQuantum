"""
Example 03 : Feature Exploration
==================================

Under the hood, FastQuantumPredictor automatically extracts two types of
structural features for each node before running the GNN:

  1. 7 heuristic centrality features  (ex: Betweenness, PageRank, Clustering)
  2. 16-step RWPE (Random Walk Positional Encodings)

This example shows how to use the low-level feature extraction API directly
for analysis or debugging, without running the full prediction pipeline.

Use case: Visualizing which nodes are structurally important for the optimizer,
or building custom pipelines that need the features as intermediate outputs.
"""

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

# --- Build a small graph ---
G = nx.karate_club_graph()  # Famous 34-node social network
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# --- Extract heuristic features (7 per node) ---
heuristics = fq.compute_node_features(G)
heuristics = np.array(heuristics)

feature_names = [
    "Degree",
    "Degree Centrality",
    "Clustering Coeff",
    "Betweenness Centrality",
    "Closeness Centrality",
    "PageRank",
    "Eigenvector Centrality",
]

print("─── Top 5 nodes by Eigenvector Centrality (most 'connected to hubs') ───")
top_nodes = np.argsort(heuristics[:, 6])[::-1][:5]
for rank, node_id in enumerate(top_nodes):
    print(f"  #{rank+1} Node {node_id:>2d} | EigCentrality={heuristics[node_id, 6]:.4f} | Degree={int(heuristics[node_id, 0])}")

# --- Extract RWPE features (16-step random walk) ---
rwpe = fq.compute_rwpe(G, k=16)
rwpe = np.array(rwpe)

print("\n─── RWPE shape ──────────────────────────────────────────────────────────")
print(f"  Matrix shape: {rwpe.shape}  (nodes × walk steps)")
print(f"  Node 0 spectral signature (first 8 steps): {rwpe[0, :8].round(4)}")
print(f"  Node 1 spectral signature (first 8 steps): {rwpe[1, :8].round(4)}")

print("\n─── Interpretation ──────────────────────────────────────────────────────")
print("  High values = node is part of a dense local neighbourhood")
print("  Low values  = node sits in a sparser or more peripheral zone")
print("  These positions encode global topology into each qubit's local state.")

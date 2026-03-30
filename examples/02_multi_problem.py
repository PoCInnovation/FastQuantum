"""
Example 02 : Multi-Problem Prediction
======================================

FastQuantum supports three combinatorial optimization problems:
  - MAXCUT      : Maximize the number of edges between two partitions
  - MIS         : Maximum Independent Set (maximize nodes with no shared edge)
  - MAX_CLIQUE  : Maximum Clique (find the largest fully-connected subgraph)

This example shows how to use the same predictor on multiple problem types
and different graph topologies.
"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

model_path = os.path.join(os.path.dirname(__file__), "..", "best_qaoa_gat_model.pt")
predictor = fq.FastQuantumPredictor(model_checkpoint=model_path)

# --- Different graph topologies ---
graphs = {
    "Erdos-Renyi (random)":        nx.erdos_renyi_graph(12, 0.4, seed=0),
    "Barabasi-Albert (scale-free)": nx.barabasi_albert_graph(12, 2, seed=0),
    "Watts-Strogatz (small-world)": nx.watts_strogatz_graph(12, 4, 0.3, seed=0),
}

problems = ["MAXCUT", "MIS", "MAX_CLIQUE"]

print("=" * 65)
print(f"{'Graph':<30} {'Problem':<12} {'γ':>10} {'β':>10}")
print("=" * 65)

for graph_name, G in graphs.items():
    for problem in problems:
        gamma, beta = predictor.predict(G, problem=problem)
        print(f"{graph_name:<30} {problem:<12} {gamma[0]:>10.4f} {beta[0]:>10.4f}")
    print("-" * 65)

"""
Example 01 : Quickstart - Warm-Starting QAOA in 3 lines
=========================================================

This is the simplest use case of the FastQuantum library.
Given any networkx graph representing a combinatorial optimization problem,
FastQuantumPredictor returns optimal QAOA angles gamma and beta
without any training or parameter tuning.

Use case: You have an optimization problem modelled as a graph.
You want to run QAOA on a quantum backend, but you need good starting
angles to avoid hundreds of variational iterations.
"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

# ---- Step 1 : Define your problem as a NetworkX graph ----
# Here: a random MaxCut problem with 15 qubits
G = nx.erdos_renyi_graph(n=15, p=0.4, seed=42)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- Step 2 : Load the predictor (pre-trained model) ----
# The model file (.pt) ships with the library.
# Inference runs in < 20ms on CPU.
model_path = os.path.join(os.path.dirname(__file__), "..", "best_qaoa_gat_model.pt")
predictor = fq.FastQuantumPredictor(model_checkpoint=model_path)

# ---- Step 3 : Predict optimal QAOA angles instantly ----
gamma, beta = predictor.predict(G, problem="MAXCUT")

print(f"\n✅ Predicted QAOA parameters:")
print(f"   γ (cost Hamiltonian phase) : {gamma}")
print(f"   β (mixing Hamiltonian)     : {beta}")

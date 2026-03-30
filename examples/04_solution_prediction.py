"""
Example 04 : Direct Solution Prediction
=========================================

FastQuantumSolutionPredictor uses a different model than FastQuantumPredictor.
Instead of predicting QAOA parameters (gamma, beta), it directly predicts
the optimal binary solution bitstring for the optimization problem.

This is a more ambitious objective: going directly from a graph to a partition
[0, 1, 0, 1, ...] without any quantum circuit execution.

Model: QuantumGraphModel (GAT Encoder → Problem Embedding → Graph Transformer → Classifier)
Checkpoint: best_model_solar-sweep-1.pt (trained on HuggingFace)
"""

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

# ---- Step 1: Define your optimization problem as a graph ----
G = nx.erdos_renyi_graph(n=12, p=0.4, seed=42)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- Step 2: Load the Solution Predictor ----
model_path = os.path.join(os.path.dirname(__file__), "..", "best_model_solar-sweep-1.pt")
predictor = fq.FastQuantumSolutionPredictor(model_checkpoint=model_path)

# ---- Step 3: Predict the solution bitstring ----
solution = predictor.predict(G, problem="MAXCUT")

print(f"\n✅ Predicted Solution Bitstring:")
print(f"   {solution}")
print(f"   Length: {len(solution)} (one per node)")

# ---- Step 4: Interpret the solution ----
partition_A = [node for node, bit in enumerate(solution) if bit == 1]
partition_B = [node for node, bit in enumerate(solution) if bit == 0]

print(f"\n📊 MaxCut Partition:")
print(f"   Partition A (bit=1): nodes {partition_A}")
print(f"   Partition B (bit=0): nodes {partition_B}")

# Count edges crossing the cut
cut_edges = [(u, v) for u, v in G.edges() if solution[u] != solution[v]]
total_edges = G.number_of_edges()
print(f"\n✂️  Cut edges: {len(cut_edges)} / {total_edges}  "
      f"({100 * len(cut_edges) / total_edges:.0f}% of all edges)")

print("\n💡 Note: This model predicts directly, without running a quantum circuit.")
print("   The solution quality depends on the training performance of the loaded checkpoint.")

"""
Example 05 : Comparison - Warm-Starting QAOA vs Direct Prediction
===================================================================

This example compares the two flagship AI approaches of FastQuantum:

1. QAOA Warm-Starting (Angles Prediction):
   - Model: FastQuantumPredictor
   - Output: Optimal angles (gamma, beta) for a quantum circuit.
   - Usage: Feed these angles to Qiskit/PennyLane to find the solution.

2. Direct Solution Prediction (End-to-End):
   - Model: FastQuantumSolutionPredictor
   - Output: The final bitstring solution [0, 1, 0, 1, ...] directly.
   - Usage: Immediate answer without needing a quantum computer.

This script runs both on the same graph and compares the results.
"""

import os
import sys
import networkx as nx
import torch

# Add parent directory to path to import fastquantum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

def compute_cut_value(G, bitstring):
    """Computes the MaxCut value for a given bitstring."""
    cut = 0
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            cut += 1
    return cut

def run_comparison():
    # 1. Setup Graph
    n_nodes = 14
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=42)
    print(f"🚀 Problem: MaxCut on {n_nodes} nodes ({G.number_of_edges()} edges)")
    print("-" * 60)

    # 2. IA Approach 1: QAOA Parameters
    print("\n[Approach 1] QAOA Parameter Prediction (Warm-Starting)")
    try:
        model_p1 = os.path.join(os.path.dirname(__file__), "..", "best_qaoa_gat_model.pt")
        predictor_angles = fq.FastQuantumPredictor(model_p1)
        gamma, beta = predictor_angles.predict(G, problem="MAXCUT")
        print(f"  ✨ Predicted Angles:")
        print(f"     γ = {gamma}")
        print(f"     β = {beta}")
        print("  💡 Usage: These angles are the optimal entry point for your QAOA circuit.")
    except Exception as e:
        print(f"  ❌ Error loading/running Predictor: {e}")

    # 3. IA Approach 2: Direct bitstring
    print("\n[Approach 2] Direct Solution Prediction (End-to-End)")
    try:
        model_p2 = os.path.join(os.path.dirname(__file__), "..", "best_model_solar-sweep-1.pt")
        predictor_sol = fq.FastQuantumSolutionPredictor(model_p2)
        bitstring = predictor_sol.predict(G, problem="MAXCUT")
        cut_val = compute_cut_value(G, bitstring)
        
        print(f"  ✨ Predicted Result: {bitstring}")
        print(f"  📈 Quality Score (MaxCut): {cut_val} edges cut")
        print(f"  📊 Performance: {100 * cut_val / G.number_of_edges():.1f}% of all edges cut.")
    except Exception as e:
        print(f"  ❌ Error loading/running SolutionPredictor: {e}")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("- Use FastQuantumPredictor if you HAVE a quantum computer (to save time/cost).")
    print("- Use FastQuantumSolutionPredictor if you DON'T have a quantum computer")
    print("  and want an instant GNN-powered heuristic.")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()

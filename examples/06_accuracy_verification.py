"""
Example 06 : Ground Truth Verification
========================================

How do we know if the IA prediction is "correct"?
For small graphs, we can use Brute Force to find the absolute mathematical
optimum (Ground Truth) and compare it with the IA's guess.

This script:
1. Generates a small graph (n=12 nodes).
2. Computes the EXACT MaxCut via Brute Force.
3. Predicts the MaxCut via FastQuantumSolutionPredictor.
4. Compares the two to calculate the Approximation Ratio.
"""

import os
import sys
import time
import networkx as nx

# Add parent directory to path to import fastquantum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fastquantum as fq

def solve_maxcut_brute_force(G):
    """Finds the exact optimal MaxCut value and bitstring."""
    n = G.number_of_nodes()
    best_val = -1
    best_x = [0] * n
    
    # Iterate through all 2^n possible partitions
    for i in range(1 << n):
        x = [(i >> bit) & 1 for bit in range(n)]
        current_cut = 0
        for u, v in G.edges():
            if x[u] != x[v]:
                current_cut += 1
        
        if current_cut > best_val:
            best_val = current_cut
            best_x = x
            
    return best_val, best_x

def run_verification():
    # 1. Setup small graph (n=12 nodes is 2^12 = 4096 combinations, very fast)
    n_nodes = 12
    G = nx.erdos_renyi_graph(n_nodes, 0.4, seed=123)
    print(f"🔍 Testing Accuracy on {n_nodes} nodes...")

    # 2. BRUTE FORCE (Ground Truth)
    print(f"-> Computing Exact Solution via Brute Force...")
    start_time = time.time()
    exact_val, exact_sol = solve_maxcut_brute_force(G)
    bf_time = time.time() - start_time
    print(f"   [DONE] Exact MaxCut: {exact_val} edges (Time: {bf_time:.4f}s)")

    # 3. IA PREDICTION
    print(f"-> Computing IA Prediction...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "best_model_solar-sweep-1.pt")
    predictor = fq.FastQuantumSolutionPredictor(model_path)
    
    start_time = time.time()
    ia_sol = predictor.predict(G, problem="MAXCUT")
    ia_time = time.time() - start_time
    
    # Calculate IA cut value
    ia_val = 0
    for u, v in G.edges():
        if ia_sol[u] != ia_sol[v]:
            ia_val += 1
    
    print(f"   [DONE] IA MaxCut: {ia_val} edges (Time: {ia_time:.4f}s)")

    # 4. RESULTS COMPARISON
    ratio = ia_val / exact_val if exact_val > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"{'METRIC':<20} | {'GROUND TRUTH':<15} | {'FASTQUANTUM IA':<15}")
    print("-" * 50)
    print(f"{'Objective Value':<20} | {exact_val:<15} | {ia_val:<15}")
    print(f"{'Execution Time':<20} | {bf_time:>14.4f}s | {ia_time:>14.4f}s")
    print("-" * 50)
    print(f"🚀 APPROXIMATION RATIO: {ratio:.2%}")
    print("=" * 50)

    if ratio >= 1.0:
        print("\n🏆 PERF PARFAITE : L'IA a trouvé l'optimum global !")
    elif ratio >= 0.90:
        print("\n🔥 EXCELLENT : L'IA est à plus de 90% de l'optimum.")
    else:
        print("\n📈 ACCEPTABLE : L'IA fournit une bonne base, mais peut être affinée.")

if __name__ == "__main__":
    run_verification()

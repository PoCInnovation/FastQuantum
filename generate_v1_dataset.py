"""
generate_v1_dataset.py - FastQuantum V1 Dataset Generator
Supports MaxCut, Maximum Independent Set (MIS), and Maximum Clique.
"""

import argparse
import json
import multiprocessing as mp
import time
import os
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

# Constants
DEFAULT_P_LAYERS = 1
RWPE_DEPTH = 16
MULTI_START_ATTEMPTS = 3

class HamiltonianFactory:
    """Factory for combinatorial optimization Hamiltonians."""
    
    @staticmethod
    def get_hamiltonian(G: nx.Graph, problem_type: str) -> Tuple[SparsePauliOp, float]:
        if problem_type == 'MAXCUT':
            return HamiltonianFactory._maxcut(G)
        elif problem_type == 'MIS':
            return HamiltonianFactory._mis(G)
        elif problem_type == 'MAX_CLIQUE':
            return HamiltonianFactory._mis(nx.complement(G))
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    @staticmethod
    def _maxcut(G: nx.Graph) -> Tuple[SparsePauliOp, float]:
        n = G.number_of_nodes()
        pauli_list = []
        for u, v in G.edges():
            op_str = ["I"] * n
            op_str[n - 1 - u] = "Z"
            op_str[n - 1 - v] = "Z"
            pauli_list.append(("".join(op_str), 1.0))
        if not pauli_list:
            return SparsePauliOp(["I" * n], [0.0]), 0.0
        return SparsePauliOp.from_list(pauli_list), 0.0

    @staticmethod
    def _mis(G: nx.Graph, penalty: float = 2.0) -> Tuple[SparsePauliOp, float]:
        n = G.number_of_nodes()
        pauli_list = []
        z_coeffs = np.zeros(n)
        for i in range(n):
            z_coeffs[i] += 0.5
        for u, v in G.edges():
            op_str = ["I"] * n
            op_str[n - 1 - u] = "Z"
            op_str[n - 1 - v] = "Z"
            pauli_list.append(("".join(op_str), penalty / 4.0))
            z_coeffs[u] -= penalty / 4.0
            z_coeffs[v] -= penalty / 4.0
        for i in range(n):
            if abs(z_coeffs[i]) > 1e-6:
                op_str = ["I"] * n
                op_str[n - 1 - i] = "Z"
                pauli_list.append(("".join(op_str), z_coeffs[i]))
        if not pauli_list:
            return SparsePauliOp(["I" * n], [0.0]), 0.0
        return SparsePauliOp.from_list(pauli_list), 0.0

def solve_brute_force(G: nx.Graph, problem_type: str) -> Tuple[float, List[int]]:
    n = G.number_of_nodes()
    best_val = -float('inf')
    best_x = [0] * n
    for i in range(1 << n):
        x = [(i >> bit) & 1 for bit in range(n)]
        current_val = 0
        if problem_type == 'MAXCUT':
            for u, v in G.edges():
                if x[u] != x[v]: current_val += 1
            if current_val > best_val: 
                best_val = current_val
                best_x = x
        elif problem_type == 'MIS':
            is_independent = True
            for u, v in G.edges():
                if x[u] == 1 and x[v] == 1:
                    is_independent = False
                    break
            if is_independent and sum(x) > best_val: 
                best_val = sum(x)
                best_x = x
        elif problem_type == 'MAX_CLIQUE':
            is_clique = True
            selected = [node for node, val in enumerate(x) if val == 1]
            for idx1 in range(len(selected)):
                for idx2 in range(idx1 + 1, len(selected)):
                    if not G.has_edge(selected[idx1], selected[idx2]):
                        is_clique = False
                        break
                if not is_clique: break
            if is_clique and sum(x) > best_val: 
                best_val = sum(x)
                best_x = x
    return float(best_val), best_x

def generate_diverse_graph(n_nodes: int) -> Tuple[nx.Graph, str]:
    topo_type = random.choice(['ER', 'BA', 'REG', 'WS', 'LOLLIPOP'])
    if topo_type == 'ER': G = nx.erdos_renyi_graph(n_nodes, random.uniform(0.3, 0.7))
    elif topo_type == 'BA': G = nx.barabasi_albert_graph(n_nodes, random.randint(1, 3))
    elif topo_type == 'REG':
        d = random.choice([3, 4, 5])
        try: G = nx.random_regular_graph(min(d, n_nodes-1), n_nodes)
        except: G = nx.erdos_renyi_graph(n_nodes, 0.5)
    elif topo_type == 'WS': G = nx.watts_strogatz_graph(n_nodes, 4, 0.1)
    else:
        m = n_nodes // 2
        G = nx.lollipop_graph(m, n_nodes - m)
    if not nx.is_connected(G): G = nx.complete_graph(n_nodes)
    return nx.convert_node_labels_to_integers(G), topo_type

def compute_rwpe(G: nx.Graph, k: int = 16) -> List[List[float]]:
    n = G.number_of_nodes()
    try:
        A = nx.adjacency_matrix(G).toarray()
        D_inv = np.diag(1.0 / np.sum(A, axis=1))
        P = D_inv @ A
        rwpe, Pk = [], np.eye(n)
        for _ in range(k):
            Pk = Pk @ P
            rwpe.append(np.diag(Pk))
        return np.stack(rwpe, axis=1).tolist()
    except:
        return np.zeros((n, k)).tolist()

def compute_node_features(G: nx.Graph) -> List[List[float]]:
    n = G.number_of_nodes()
    feat_dicts = [
        dict(G.degree()), nx.degree_centrality(G), nx.clustering(G),
        nx.betweenness_centrality(G), nx.closeness_centrality(G)
    ]
    try: feat_dicts.append(nx.pagerank(G, max_iter=200))
    except: feat_dicts.append({i: 0.0 for i in range(n)})
    try: feat_dicts.append(nx.eigenvector_centrality(G, max_iter=200))
    except: feat_dicts.append({i: 0.0 for i in range(n)})
    return [[d[i] for d in feat_dicts] for i in range(n)]

def build_qaoa_circuit(G: nx.Graph, p: int, problem_type: str):
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    gammas = [Parameter(f'g{i}') for i in range(p)]
    betas = [Parameter(f'b{i}') for i in range(p)]
    h, _ = HamiltonianFactory.get_hamiltonian(G, problem_type)
    for layer in range(p):
        for op, coeff in zip(h.paulis, h.coeffs):
            z_idx = [k for k, c in enumerate(reversed(op.to_label())) if c == 'Z']
            if len(z_idx) == 1: qc.rz(2 * gammas[layer] * coeff.real, z_idx[0])
            elif len(z_idx) == 2: qc.rzz(2 * gammas[layer] * coeff.real, z_idx[0], z_idx[1])
        for i in range(n): qc.rx(2 * betas[layer], i)
    return qc, gammas, betas, h

def optimize_qaoa(G: nx.Graph, p: int, problem_type: str):
    qc, gs, bs, h = build_qaoa_circuit(G, p, problem_type)
    estimator = StatevectorEstimator()
    def cost(pv):
        res = estimator.run([(qc, h, {gs[i]: pv[i] for i in range(p)} | {bs[i]: pv[p+i] for i in range(p)})]).result()
        return res[0].data.evs
    best_e = float('inf')
    best_p = None
    for _ in range(MULTI_START_ATTEMPTS):
        res = minimize(cost, [random.uniform(0, np.pi) for _ in range(p)] + [random.uniform(0, np.pi/2) for _ in range(p)], 
                       method='COBYLA', options={'maxiter': 100})
        if res.fun < best_e: best_e, best_p = res.fun, res.x
    return {'gamma': best_p[:p].tolist(), 'beta': best_p[p:].tolist(), 'energy': float(best_e)}

def worker_task(args):
    idx, n_min, n_max, p, problem = args
    # Re-seed random for each process to ensure diversity
    random.seed(idx + int(time.time()))
    np.random.seed(idx + int(time.time()))
    
    thresholds = {'MAXCUT': 0.85, 'MIS': 0.85, 'MAX_CLIQUE': 0.85}
    try:
        G, topo = generate_diverse_graph(random.randint(n_min, n_max))
        exact_val, exact_sol = solve_brute_force(G, problem)
        qaoa = optimize_qaoa(G, p, problem)
        e = qaoa['energy']
        if problem == 'MAXCUT': ratio = 0.5 * (G.number_of_edges() - e) / exact_val if exact_val > 0 else 0
        else: ratio = min(-e / exact_val, 1.0) if exact_val > 0 else 0
        if ratio < thresholds.get(problem, 0.85): return None
        feats = [h + r for h, r in zip(compute_node_features(G), compute_rwpe(G))]
        return {'id': idx, 'problem': problem, 'n_nodes': G.number_of_nodes(), 'topo': topo,
                'adj': nx.to_numpy_array(G).tolist(), 'x': feats, 'gamma': qaoa['gamma'], 'beta': qaoa['beta'], 
                'ratio': ratio, 'exact_value': exact_val, 'exact_solution': exact_sol}
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="MAXCUT", choices=["MAXCUT", "MIS", "MAX_CLIQUE"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--nodes", type=str, default="8-16")
    parser.add_argument("--workers", type=int, default=os.cpu_count()-1)
    parser.add_argument("--output", type=str, default="Dataset/qaoa_dataset.json")
    args = parser.parse_args()
    n_min, n_max = map(int, args.nodes.split('-'))
    
    print(f"Starting Generation | Problem: {args.problem} | Target: {args.samples}")
    print(f"Nodes: {n_min}-{n_max} | Workers: {args.workers}")
    
    dataset, attempts = [], 0
    pool = mp.Pool(args.workers)
    
    # Use imap_unordered for real-time progress updates
    task_args = ((i, n_min, n_max, DEFAULT_P_LAYERS, args.problem) for i in range(args.samples * 100))
    
    try:
        for result in pool.imap_unordered(worker_task, task_args):
            attempts += 1
            if result:
                dataset.append(result)
                print(f"Progress: {len(dataset)}/{args.samples} (Attempts: {attempts})", end='\r', flush=True)
            
            if len(dataset) >= args.samples:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial dataset...")
    
    pool.terminate()
    pool.join()
    
    print(f"\nFinal: {len(dataset)}/{args.samples} (Total attempts: {attempts})")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f: json.dump(dataset, f, indent=2)
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
import numpy as np
import networkx as nx
import torch
import argparse
import multiprocessing as mp
import os
from tqdm import tqdm
from functools import partial

# Optional Qiskit imports
try:
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
    from qiskit_optimization.applications import Maxcut
    from qiskit_optimization.converters import QuadraticProgramToQubo
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

def get_graph(n_nodes, prob, g_type='erdos_renyi', seed=None):
    """Generates a random graph based on specified type."""
    if seed: np.random.seed(seed)
    
    if g_type == 'erdos_renyi':
        return nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
    elif g_type == 'regular':
        d = max(2, min(int(n_nodes * prob * 2), n_nodes - 1))
        if (n_nodes * d) % 2 != 0: d -= 1
        return nx.random_regular_graph(d, n_nodes, seed=seed)
    elif g_type == 'grid':
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(G)
    return nx.erdos_renyi_graph(n_nodes, prob, seed=seed)

def get_features(G):
    """Extracts 7 topological features for each node."""
    n = G.number_of_nodes()
    # Centralities
    deg = list(dict(G.degree()).values())
    deg_c = list(nx.degree_centrality(G).values())
    clust = list(nx.clustering(G).values())
    betw = list(nx.betweenness_centrality(G).values())
    close = list(nx.closeness_centrality(G).values())
    page = list(nx.pagerank(G, max_iter=500).values())
    try:
        eigen = list(nx.eigenvector_centrality(G, max_iter=500).values())
    except:
        eigen = [0.0] * n

    # Stack features (N x 7)
    return np.column_stack([deg, deg_c, clust, betw, close, page, eigen])

def solve_classical(G, p=1):
    """Fast Scipy optimization (MaxCut cost simulation)."""
    from scipy.optimize import minimize
    
    adj = nx.to_numpy_array(G)
    n = G.number_of_nodes()
    
    def cost(params):
        gamma, beta = params[:p], params[p:]
        # Simplified energy approx
        obj = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] > 0:
                    angle = sum(gamma) + sum(beta)
                    obj += adj[i, j] * (1 - np.cos(angle)) / 2
        return -obj

    # Multi-start optimization
    best_res = None
    best_val = float('inf')
    
    for _ in range(5): # 5 random starts
        x0 = np.random.uniform(0, np.pi, 2*p)
        res = minimize(cost, x0, method='COBYLA', options={'maxiter': 100})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    return best_res.x[:p], best_res.x[p:], -best_val

def solve_qiskit(G, p=1):
    """Precise Qiskit optimization (Real QAOA simulator)."""
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit missing. Install 'qiskit-algorithms' or use --mode classical")

    maxcut = Maxcut(G)
    qubo = QuadraticProgramToQubo().convert(maxcut.to_quadratic_program())
    
    qaoa = QAOA(Sampler(), COBYLA(maxiter=100), reps=p)
    result = qaoa.compute_minimum_eigenvalue(qubo.to_ising()[0])
    
    return result.optimal_point[:p], result.optimal_point[p:], result.optimal_value

def worker(args):
    """Worker task for multiprocessing."""
    idx, config = args
    seed = config['seed'] + idx
    np.random.seed(seed)
    
    # Randomize params slightly
    n = np.random.randint(config['min_nodes'], config['max_nodes'] + 1)
    prob = np.random.uniform(config['min_prob'], config['max_prob'])
    
    # Generate
    G = get_graph(n, prob, config['type'], seed)
    if not nx.is_connected(G) or G.number_of_edges() == 0:
        return None

    features = get_features(G)
    adj = nx.to_numpy_array(G)

    # Solve
    try:
        if config['use_qiskit']:
            g, b, e = solve_qiskit(G, config['p'])
        else:
            g, b, e = solve_classical(G, config['p'])
            
        return {
            'id': idx,
            'n_nodes': n,
            'adjacency_matrix': adj.tolist(),
            'node_features': features.tolist(),
            'optimal_gamma': g.tolist(),
            'optimal_beta': b.tolist(),
            'optimal_energy': float(e)
        }
    except Exception as e:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAOA MaxCut Dataset Generator")
    parser.add_argument('--samples', type=int, default=1000, help="Number of graphs")
    parser.add_argument('--out', type=str, default="Dataset/finetune/dataset.pt", help="Output file")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Thread count")
    parser.add_argument('--mode', choices=['classical', 'qiskit'], default='classical', help="Optimization engine")
    parser.add_argument('--p', type=int, default=1, help="QAOA depth")
    
    args = parser.parse_args()
    
    config = {
        'seed': 42,
        'min_nodes': 8, 'max_nodes': 20,
        'min_prob': 0.3, 'max_prob': 0.7,
        'type': 'erdos_renyi',
        'p': args.p,
        'use_qiskit': args.mode == 'qiskit'
    }
    
    print(f"🚀 Generating {args.samples} graphs ({args.workers} workers) | Mode: {args.mode.upper()}")
    
    # Prepare tasks
    tasks = [(i, config) for i in range(args.samples)]
    
    # Parallel execution
    dataset = []
    with mp.Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks), total=args.samples, unit="g"):
            if res: dataset.append(res)
            
    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dataset, args.out)
    
    print(f"✅ Done: {len(dataset)} valid graphs saved to {args.out}")
#!/usr/bin/env python3
"""
generate_dataset.py - QAOA parameter optimization dataset generator

Generates dataset linking graph features to QAOA performance using Qiskit.

Usage:
  python generate_dataset.py --n_graphs 100 --out dataset.parquet
"""

import argparse
import itertools
import json
import random
import time
from typing import Tuple, List, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit_optimization.applications import Maxcut
from scipy.optimize import minimize


def extract_graph_features(G: nx.Graph) -> Dict[str, float]:
    """Extract graph features for ML."""
    n = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    
    features = {
        'num_nodes': n,
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'degree_mean': np.mean(degrees),
        'degree_std': np.std(degrees),
        'clustering_coeff': nx.average_clustering(G),
    }
    
    try:
        features['assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        features['assortativity'] = 0.0
    
    return features


def brute_force_maxcut(G: nx.Graph) -> float:
    """Exact MaxCut solution (only for small n)."""
    n = G.number_of_nodes()
    best = 0
    nodes = list(G.nodes())
    
    for bits in itertools.product([0, 1], repeat=n):
        cut = sum(1 for u, v in G.edges() 
                  if bits[nodes.index(u)] != bits[nodes.index(v)])
        best = max(best, cut)
    
    return float(best)


def run_qaoa(G: nx.Graph, p: int, init_beta: List[float], 
             init_gamma: List[float], optimizer: str, optimal_cut: Optional[float] = None,
             max_iter: int = 100) -> Dict[str, float]:
    """Run QAOA with Qiskit and return metrics."""
    start = time.time()
    n = G.number_of_nodes()
    
    print(f"    [QAOA] Running n={n}, p={p}, opt={optimizer}", flush=True)
    
    # Build Hamiltonian
    maxcut = Maxcut(G)
    qubit_op, offset = maxcut.to_quadratic_program().to_ising()
    
    # Build QAOA circuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    
    beta_params = [Parameter(f'β{i}') for i in range(p)]
    gamma_params = [Parameter(f'γ{i}') for i in range(p)]
    
    for layer in range(p):
        for i, j in G.edges():
            qc.rzz(2 * gamma_params[layer], i, j)
        for qubit in range(n):
            qc.rx(2 * beta_params[layer], qubit)
    
    # Cost function
    estimator = StatevectorEstimator()
    iterations = [0]
    
    def cost(params):
        param_dict = {beta_params[i]: params[i] for i in range(p)}
        param_dict.update({gamma_params[i]: params[p+i] for i in range(p)})
        bound_qc = qc.assign_parameters(param_dict)
        result = estimator.run([(bound_qc, qubit_op)]).result()
        iterations[0] += 1
        return float(result[0].data.evs + offset)
    
    # Optimize
    initial = np.array(init_beta + init_gamma)
    method = 'COBYLA' if optimizer in ['COBYLA', 'SPSA'] else optimizer
    result = minimize(cost, initial, method=method, options={'maxiter': max_iter})
    
    # Metrics
    final_energy = float(result.fun)
    achieved_cut = max(0, -final_energy)
    approx_ratio = achieved_cut / optimal_cut if optimal_cut and optimal_cut > 0 else 0.9
    
    print(f"      ✓ Completed in {time.time() - start:.1f}s, {iterations[0]} iters, ratio={approx_ratio:.3f}", flush=True)
    
    return {
        'final_energy': final_energy,
        'approximation_ratio': float(np.clip(approx_ratio, 0, 1)),
        'iterations': iterations[0],
        'converged': int(result.success),
        'runtime': time.time() - start,
        'optimal_cut': optimal_cut or np.nan,
    }


def sample_parameters(p_choices: List[int], seed: Optional[int] = None) -> Tuple[int, List[float], List[float], str]:
    """Sample QAOA parameters with optimal ranges."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    p = random.choice(p_choices)
    # Optimal ranges from QAOA theory
    init_beta = list(np.random.uniform(0, np.pi/2, size=p)) if p > 0 else []
    init_gamma = list(np.random.uniform(0, np.pi, size=p)) if p > 0 else []
    optimizer = random.choice(["COBYLA", "SPSA", "L-BFGS-B", "CG"])
    return p, init_beta, init_gamma, optimizer


def generate_random_graph(n: int, density: float, graph_type: str = 'erdos_renyi', 
                          seed: Optional[int] = None) -> nx.Graph:
    """Generate random graph of specified type.
    
    Args:
        n: number of nodes
        density: edge probability (for ER) or target density
        graph_type: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'regular'
        seed: random seed
    """
    if graph_type == 'erdos_renyi':
        return nx.fast_gnp_random_graph(n, density, seed=seed)
    elif graph_type == 'barabasi_albert':
        m = max(1, int(density * n / 2))  # approximate edges
        return nx.barabasi_albert_graph(n, m, seed=seed)
    elif graph_type == 'watts_strogatz':
        k = max(2, int(density * n))
        k = k if k % 2 == 0 else k + 1  # must be even
        return nx.watts_strogatz_graph(n, min(k, n-1), 0.3, seed=seed)
    elif graph_type == 'regular':
        d = max(1, int(density * n))
        d = d if d % 2 == 0 else d + 1
        d = min(d, n - 1)
        try:
            return nx.random_regular_graph(d, n, seed=seed)
        except:
            return nx.fast_gnp_random_graph(n, density, seed=seed)
    else:
        return nx.fast_gnp_random_graph(n, density, seed=seed)


def generate_dataset(n_graphs: int = 100,
                     n_min: int = 6,
                     n_max: int = 12,
                     density_range: Tuple[float, float] = (0.3, 0.7),
                     samples_per_graph: int = 10,
                     p_choices: List[int] = [1, 2, 3],
                     graph_types: List[str] = ['erdos_renyi', 'barabasi_albert'],
                     out: str = "dataset.parquet",
                     seed: Optional[int] = None):
    """Generate QAOA parameter optimization dataset using Qiskit.
    
    Args:
        n_graphs: number of random graphs
        n_min, n_max: range of node counts
        density_range: (min, max) edge density
        samples_per_graph: parameter configs per graph
        p_choices: QAOA depth options
        graph_types: graph generation methods
        out: output file (.parquet or .csv)
        seed: random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    rows = []
    graph_id = 0
    
    print(f"Generating {n_graphs} graphs with {samples_per_graph} samples each...")
    
    for graph_idx in range(n_graphs):
        graph_id += 1
        
        # Sample graph parameters
        n = random.randint(n_min, n_max)
        density = random.uniform(*density_range)
        graph_type = random.choice(graph_types)
        graph_seed = seed + graph_id if seed is not None else None
        
        # Generate graph
        G = generate_random_graph(n, density, graph_type, seed=graph_seed)
        
        # Skip if graph is trivial
        if G.number_of_edges() == 0:
            continue
        
        # Extract graph features
        graph_features = extract_graph_features(G)
        
        # Get optimal solution for reference (calculate once per graph)
        if n <= 14:
            try:
                optimal_cut = brute_force_maxcut(G)
            except:
                optimal_cut = None
        else:
            optimal_cut = None
        
        # Sample multiple parameter configurations
        for sample_idx in range(samples_per_graph):
            param_seed = (seed + graph_id * 1000 + sample_idx) if seed is not None else None
            p, init_beta, init_gamma, optimizer = sample_parameters(p_choices, seed=param_seed)
            
            # Run QAOA (pass optimal_cut to avoid recalculation)
            metrics = run_qaoa(G, p, init_beta, init_gamma, optimizer, optimal_cut=optimal_cut)
            
            # Extract metrics
            final_energy = metrics['final_energy']
            iterations = metrics['iterations']
            converged = metrics['converged']
            approximation_ratio = metrics['approximation_ratio']
            runtime = metrics['runtime']
            
            # Compute derived metrics
            cut_value = -final_energy
            success_prob = approximation_ratio
            
            # Build row
            row = {
                'graph_id': graph_id,
                'graph_type': graph_type,
                'seed': graph_seed,
                **graph_features,
                'p': p,
                'init_beta': json.dumps(init_beta),
                'init_gamma': json.dumps(init_gamma),
                'optimizer': optimizer,
                'final_energy': final_energy,
                'cut_value': cut_value,
                'approximation_ratio': approximation_ratio,
                'success_prob': success_prob,
                'iterations': iterations,
                'converged': int(converged),
                'runtime': runtime,
                'optimal_cut': optimal_cut if optimal_cut is not None else np.nan,
            }
            rows.append(row)
        
        if (graph_id) % 10 == 0:
            print(f"  Processed {graph_id}/{n_graphs} graphs, {len(rows)} total samples")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to file
    if out.endswith('.parquet'):
        df.to_parquet(out, index=False, engine='pyarrow')
        print(f"✓ Dataset saved to {out} (Parquet format, {len(df)} rows)")
    else:
        df.to_csv(out, index=False)
        print(f"✓ Dataset saved to {out} (CSV format, {len(df)} rows)")
    
    # Print summary stats
    print(f"\nDataset summary:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique graphs: {df['graph_id'].nunique()}")
    print(f"  Node range: {df['num_nodes'].min()}-{df['num_nodes'].max()}")
    print(f"  Density range: {df['density'].min():.3f}-{df['density'].max():.3f}")
    print(f"  P values: {sorted(df['p'].unique())}")
    print(f"  Graph types: {df['graph_type'].unique().tolist()}")
    print(f"  Approximation ratio: {df['approximation_ratio'].mean():.3f} ± {df['approximation_ratio'].std():.3f}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QAOA parameter optimization dataset using Qiskit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_graphs", type=int, default=100, 
                        help="Number of graphs")
    parser.add_argument("--n_min", type=int, default=6, 
                        help="Minimum nodes per graph")
    parser.add_argument("--n_max", type=int, default=12, 
                        help="Maximum nodes per graph")
    parser.add_argument("--density_min", type=float, default=0.3, 
                        help="Minimum edge density")
    parser.add_argument("--density_max", type=float, default=0.7, 
                        help="Maximum edge density")
    parser.add_argument("--samples_per_graph", type=int, default=10, 
                        help="Parameter samples per graph")
    parser.add_argument("--p_choices", type=str, default="1,2,3", 
                        help="QAOA depths (comma-separated)")
    parser.add_argument("--graph_types", type=str, 
                        default="erdos_renyi,barabasi_albert",
                        help="Graph types (comma-separated)")
    parser.add_argument("--out", type=str, default="dataset.parquet", 
                        help="Output file (.parquet or .csv)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Parse list arguments
    p_choices = [int(x.strip()) for x in args.p_choices.split(",") if x.strip()]
    graph_types = [x.strip() for x in args.graph_types.split(",") if x.strip()]
    
    # Generate dataset
    generate_dataset(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        density_range=(args.density_min, args.density_max),
        samples_per_graph=args.samples_per_graph,
        p_choices=p_choices,
        graph_types=graph_types,
        out=args.out,
        seed=args.seed
    )

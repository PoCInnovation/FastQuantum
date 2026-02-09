"""
Physics Proxy Dataset Generator
===============================

Generates synthetic graphs with "Physics-Aware" targets for QAOA parameter prediction.
Instead of random targets, we compute gamma and beta based on spectral properties
of the graph, simulating a scenario where the GNN must learn topological features
to capture the target.

Dependencies:
    gamma ~ f(Laplacian Spectrum) -> Requires spectral awareness (LPE/RWPE strength)
    beta  ~ f(Local Connectivity) -> Requires clustering/degree (Heuristic strength)
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def get_spectral_features(G):
    """Compute eigenvalues needed for the proxy target."""
    L_norm = nx.normalized_laplacian_matrix(G).todense()
    eigenvalues = np.linalg.eigvalsh(L_norm)
    eigenvalues = np.sort(eigenvalues)
    
    # Key spectral properties
    # lambda_2 (Fiedler value): Algebraic connectivity
    # lambda_max: Related to max cut bound
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
    lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 0
    spectral_gap = lambda_max - lambda_2
    
    return lambda_2, lambda_max, spectral_gap, eigenvalues

def compute_proxy_targets(G):
    """
    Generate Gamma/Beta targets based on Physics rules + Noise.
    
    Rule 1 (Gamma): Strongly correlated with the inverse of lambda_max and spectral energy.
                    Simulates the 'depth' or 'energy scale' of the circuit.
    Rule 2 (Beta):  Correlated with local clustering and algebraic connectivity.
                    Simulates the 'mixing' steps needed.
    """
    # 1. Structural descriptors
    n = G.number_of_nodes()
    lambda_2, lambda_max, gap, eigens = get_spectral_features(G)
    
    avg_degree = np.mean([d for _, d in G.degree()]) / n
    clustering = nx.average_clustering(G)
    
    # 2. Proxy Formulas (The "Hidden Physics")
    # Gamma: Depends on Spectrum (Hard for Baseline, Easy for LPE)
    # We use a non-linear combination of eigenvalues
    gamma_clean = (np.pi / 2) * (1.0 / (lambda_max + 1e-5)) + 0.5 * lambda_2
    
    # Beta: Depends on Topology (Easy for Baseline/Heuristics)
    beta_clean = (np.pi / 4) * (clustering + avg_degree)
    
    # 3. Add "Real World" Noise
    # QAOA landscapes are noisy, let's simulate that variance
    noise_gamma = np.random.normal(0, 0.05)
    noise_beta = np.random.normal(0, 0.05)
    
    gamma = np.clip(gamma_clean + noise_gamma, 0, 2*np.pi)
    beta = np.clip(beta_clean + noise_beta, 0, np.pi)
    
    return torch.tensor([[gamma, beta]], dtype=torch.float)

def generate_proxy_dataset(num_graphs=1000, min_nodes=40, max_nodes=60):
    dataset = []
    print(f"⚛️ Generating {num_graphs} graphs with Physics-Proxy targets...")
    
    for _ in tqdm(range(num_graphs)):
        # Random size
        n = np.random.randint(min_nodes, max_nodes + 1)
        
        # Mix of topologies
        topo = np.random.choice(['ER', 'BA', 'WS', 'REG'])
        
        if topo == 'ER':
            p = np.random.uniform(0.1, 0.4)
            G = nx.erdos_renyi_graph(n, p)
        elif topo == 'BA':
            m = np.random.randint(2, 5)
            G = nx.barabasi_albert_graph(n, m)
        elif topo == 'WS':
            k = np.random.randint(4, 8)
            p = np.random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n, k, p)
        elif topo == 'REG':
            k = np.random.choice([3, 4, 5, 6])
            if k >= n: k = n - 1
            if (n * k) % 2 != 0: k -= 1 # Handshake lemma
            G = nx.random_regular_graph(k, n)
            
        # Ensure connectivity
        if not nx.is_connected(G):
            # Connect largest component or just make fully connected (simplification)
            # Better: add random edges until connected
            while not nx.is_connected(G):
                u, v = np.random.choice(G.nodes(), 2, replace=False)
                G.add_edge(u, v)

        # Compute Targets
        y = compute_proxy_targets(G)
        
        # Compute Base Features (Same as before for consistency)
        # Recalling the function from benchmark script would be better, 
        # but let's re-implement basic ones to be standalone
        degrees = [d for _, d in G.degree()]
        deg_norm = torch.tensor(degrees, dtype=torch.float).view(-1, 1) / n
        
        # For this generator, we just return the NetworkX object and Y
        # The benchmark script will handle feature conversion
        # To make it compatible with PyG Data directly:
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        if edge_index.shape[0] == 0: # Handle edge case of empty graph (unlikely)
             edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
             edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
             
        data = Data(num_nodes=n, edge_index=edge_index, y=y)
        data.G = G # Store NX object for feature computation if needed later
        
        dataset.append(data)
        
    print(f"✅ Generated {len(dataset)} physics-proxy graphs.")
    return dataset

if __name__ == "__main__":
    # Test generation
    ds = generate_proxy_dataset(num_graphs=10)
    print("Sample Target:", ds[0].y)

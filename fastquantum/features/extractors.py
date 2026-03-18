import numpy as np
import networkx as nx
from typing import List

def compute_rwpe(G: nx.Graph, k: int = 16) -> List[List[float]]:
    """
    Compute Random Walk Positional Encodings (RWPE) for a given graph.
    
    Args:
        G (nx.Graph): The input graph.
        k (int): Number of random walk steps.
        
    Returns:
        List[List[float]]: A list of feature vectors (size k) for each node.
    """
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
    except Exception:
        # Fallback to zero vectors in case of disconnected or empty graphs
        return np.zeros((n, k)).tolist()

def compute_node_features(G: nx.Graph) -> List[List[float]]:
    """
    Compute heuristic node features for graph nodes.
    Currently computes 7 centrality/structural features:
    Degree, Degree Centrality, Clustering Coefficient, Betweenness Centrality, 
    Closeness Centrality, PageRank, and Eigenvector Centrality.
    
    Args:
        G (nx.Graph): The input graph.
        
    Returns:
        List[List[float]]: A list of feature vectors (size 7) for each node.
    """
    n = G.number_of_nodes()
    feat_dicts = [
        dict(G.degree()), 
        nx.degree_centrality(G), 
        nx.clustering(G),
        nx.betweenness_centrality(G), 
        nx.closeness_centrality(G)
    ]
    
    try: 
        feat_dicts.append(nx.pagerank(G, max_iter=200))
    except Exception: 
        feat_dicts.append({i: 0.0 for i in range(n)})
        
    try: 
        feat_dicts.append(nx.eigenvector_centrality(G, max_iter=200))
    except Exception: 
        feat_dicts.append({i: 0.0 for i in range(n)})
        
    # Combine dictionary values into a list of feature vectors for each node order
    return [[d.get(i, d.get(str(i), 0.0)) for d in feat_dicts] for i in range(n)]

def preprocess_graph(G: nx.Graph, rwpe_steps: int = 16):
    """
    Extracts features for a given networkx graph and formats them for the model.
    Will be updated to return torch_geometric Data objects.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    
    # Ensure nodes are integers from 0 to N-1
    G = nx.convert_node_labels_to_integers(G)
    
    # Compute features
    node_feats = compute_node_features(G)
    rwpe_feats = compute_rwpe(G, k=rwpe_steps)
    
    # Concatenate features
    x = [h + r for h, r in zip(node_feats, rwpe_feats)]
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Convert PyG Data
    pyg_data = from_networkx(G)
    pyg_data.x = x_tensor
    
    return pyg_data

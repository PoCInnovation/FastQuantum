"""
Graph Memory Module (RAG)
=========================

Implements a Retrieval-Augmented Generation (RAG) system for graph parameter prediction.
Stores graph "fingerprints" in a vector database (FAISS) and retrieves optimal parameters
from similar graphs seen in the past.

Author: FastQuantum Project
"""

import numpy as np
import networkx as nx
import faiss
import torch
from .circuit_registry import CircuitRegistry # Optional, for future use

class GraphMemory:
    def __init__(self, fingerprint_dim=29, index_type='flat'):
        """
        Initialize the Graph Memory system.
        
        Args:
            fingerprint_dim (int): Dimension of the graph embedding vector.
            index_type (str): Type of FAISS index ('flat' for exact search).
        """
        self.dim = fingerprint_dim
        self.index = faiss.IndexFlatIP(self.dim) # Inner Product (Cosine sim if normalized)
        self.targets = [] # Stores (gamma, beta) for each indexed graph
        self.is_fitted = False
        
    def _compute_heuristics(self, G):
        """Compute basic node-level heuristics."""
        n = G.number_of_nodes()
        
        # Degree (normalized)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        d_norm = np.array([degrees[i] / max_deg for i in range(n)])
        
        # Clustering
        clust = np.array(list(nx.clustering(G).values()))
        
        # Centrality (Betweenness) - approx for speed
        betw = np.array(list(nx.betweenness_centrality(G, k=min(n, 20)).values()))
        
        return np.stack([d_norm, clust, betw], axis=1)

    def compute_fingerprint(self, G):
        """
        Compute the 'DNA' of a graph for similarity search.
        Combines spectral properties and aggregated heuristics.
        """
        # 1. Spectral Features (Global)
        try:
            L_norm = nx.normalized_laplacian_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(L_norm)
            eigenvalues = np.sort(eigenvalues)
            
            lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
            lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 0
            gap = lambda_max - lambda_2
            
            # Top-k small eigenvalues (padding if needed)
            k_eig = 5
            if len(eigenvalues) < k_eig:
                eigens = np.pad(eigenvalues, (0, k_eig - len(eigenvalues)))
            else:
                eigens = eigenvalues[:k_eig]
                
        except Exception:
            # Fallback if spectral fails
            lambda_2, lambda_max, gap = 0, 0, 0
            eigens = np.zeros(5)

        # 2. Heuristic Stats (Aggregated)
        node_feats = self._compute_heuristics(G)
        feat_mean = np.mean(node_feats, axis=0) # 3 dims
        feat_std = np.std(node_feats, axis=0)   # 3 dims
        feat_max = np.max(node_feats, axis=0)   # 3 dims
        
        # 3. Graph Info
        density = nx.density(G)
        
        # Combine: 
        # 3*3 (heuristics) + 5 (spectral) + 3 (spectral stats) + 1 (density) = 18 dims
        # Padding to match requested dim (29 from prototype was higher due to more heuristics)
        
        # Let's stick to the high-performance prototype logic if we can re-implement fully,
        # but for this module, let's keep it robust.
        # Vector: [Mean(3), Std(3), Max(3), Eigens(5), L2, Lmax, Gap, Density] = 18 dims
        
        fingerprint = np.concatenate([
            feat_mean, feat_std, feat_max,
            eigens,
            np.array([lambda_2, lambda_max, gap, density])
        ])
        
        # Pad with zeros if dim > current size (to allow future expansion)
        current_dim = fingerprint.shape[0]
        if current_dim < self.dim:
            fingerprint = np.pad(fingerprint, (0, self.dim - current_dim))
        elif current_dim > self.dim:
            fingerprint = fingerprint[:self.dim]
            
        return fingerprint.astype(np.float32)

    def add_graphs(self, graphs, targets):
        """
        Add a batch of graphs and their optimal parameters to the memory.
        
        Args:
            graphs (list): List of NetworkX graphs.
            targets (list/np.array): List of (gamma, beta) tuples.
        """
        fingerprints = []
        valid_targets = []
        
        for G, target in zip(graphs, targets):
            fp = self.compute_fingerprint(G)
            fingerprints.append(fp)
            valid_targets.append(target)
            
        fingerprints = np.array(fingerprints)
        faiss.normalize_L2(fingerprints)
        
        self.index.add(fingerprints)
        self.targets.extend(valid_targets)
        self.is_fitted = True
        
        return len(fingerprints)

    def retrieve(self, G, k=5):
        """
        Retrieve parameters from the k most similar graphs.
        
        Returns:
            predicted_params (np.array): Average of neighbors' parameters.
            distances (np.array): Similarity scores.
        """
        if not self.is_fitted:
            raise ValueError("GraphMemory is empty! Call add_graphs() first.")
            
        query_fp = self.compute_fingerprint(G).reshape(1, -1)
        faiss.normalize_L2(query_fp)
        
        D, I = self.index.search(query_fp, k)
        
        neighbor_indices = I[0]
        neighbor_targets = np.array([self.targets[i] for i in neighbor_indices])
        
        # Simple average prediction
        predicted_params = np.mean(neighbor_targets, axis=0)
        
        return predicted_params, D[0]

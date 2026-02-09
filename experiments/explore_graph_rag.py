"""
Graph RAG Prototype
===================

Explores the feasibility of Retrieval-Augmented Generation (RAG) for QAOA parameter prediction.
Instead of training a global model, we:
1. Compute a "Fingerprint" for each graph (Heuristics + Spectral).
2. Store these fingerprints in a FAISS Vector Database.
3. For a new graph, retrieve the most similar training graphs.
4. Predict parameters by averaging the neighbors' targets.

"""

import numpy as np
import networkx as nx
import torch
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import existing tools
from generate_physics_proxy import generate_proxy_dataset, get_spectral_features
from benchmark_encoding import compute_heuristic_features

def compute_graph_fingerprint(G):
    """
    Creates a fixed-size vector representation (embedding) for a graph.
    Combines:
    1. Aggregated Heuristic Stats (Mean/Max/Std of node features)
    2. Spectral Features (Eigenvalues)
    """
    # 1. Node-level Heuristics (7 dims per node) -> Aggregate to Graph-level
    node_feats = compute_heuristic_features(G) # Shape: (N, 7)
    
    # Aggregators: Mean, Std, Max, Min
    feat_mean = np.mean(node_feats, axis=0)
    feat_std = np.std(node_feats, axis=0)
    feat_max = np.max(node_feats, axis=0)
    
    # 2. Spectral Features (Global)
    # get_spectral_features returns: lambda_2, lambda_max, gap, eigens
    lambda_2, lambda_max, gap, eigens = get_spectral_features(G)
    
    # Use top k eigenvalues as features (padding if necessary)
    k_eig = 5
    if len(eigens) < k_eig:
        eigens = np.pad(eigens, (0, k_eig - len(eigens)))
    spectral_fingerprint = eigens[:k_eig] # Smallest k eigenvalues
    
    # Combine all into a single vector
    # 7*3 (heuristics) + 5 (spectral) + 3 (explicit spectral stats) = 29 dimensions
    fingerprint = np.concatenate([
        feat_mean, feat_std, feat_max,
        spectral_fingerprint,
        np.array([lambda_2, lambda_max, gap])
    ])
    
    return fingerprint.astype(np.float32)

def build_vector_database(dataset):
    """
    Builds a FAISS index from a list of PyG Data objects.
    """
    print("🏗️  Building Vector Database...")
    
    fingerprints = []
    targets = []
    
    for data in tqdm(dataset):
        fp = compute_graph_fingerprint(data.G)
        fingerprints.append(fp)
        targets.append(data.y.numpy().flatten())
        
    fingerprints = np.array(fingerprints)
    targets = np.array(targets)
    
    # Normalize vectors for Cosine Similarity (optional but recommended)
    faiss.normalize_L2(fingerprints)
    
    # Create Index
    d = fingerprints.shape[1] # Dimension
    index = faiss.IndexFlatIP(d) # Inner Product (Cosine sim if normalized)
    index.add(fingerprints)
    
    print(f"✅ Indexed {index.ntotal} graphs. Vector dim: {d}")
    return index, targets, fingerprints

def retrieve_and_predict(index, db_targets, query_graphs, k=5):
    """
    Retrieves k-NN for query graphs and predicts via averaging.
    """
    print(f"🔍 Querying {len(query_graphs)} graphs (k={k})...")
    
    predictions = []
    ground_truth = []
    
    for data in query_graphs:
        # Compute fingerprint
        query_fp = compute_graph_fingerprint(data.G).reshape(1, -1)
        faiss.normalize_L2(query_fp)
        
        # Search
        D, I = index.search(query_fp, k)
        
        # Retrieve neighbors' targets
        neighbor_indices = I[0]
        neighbor_targets = db_targets[neighbor_indices]
        
        # Predict: Weighted average based on similarity scores (D) could be better,
        # but simple average is a good baseline.
        pred = np.mean(neighbor_targets, axis=0)
        
        predictions.append(pred)
        ground_truth.append(data.y.numpy().flatten())
        
    return np.array(predictions), np.array(ground_truth)

def run_experiment():
    # 1. Generate Data
    # Train = Database (Knowledge Base)
    # Test = Queries
    print("Generating Database (Train) and Query (Test) sets...")
    # Using 1000 for DB to have good coverage, 100 for testing
    full_dataset = generate_proxy_dataset(num_graphs=1100, min_nodes=40, max_nodes=60)
    
    db_dataset = full_dataset[:1000]
    query_dataset = full_dataset[1000:]
    
    # 2. Build DB
    index, db_targets, _ = build_vector_database(db_dataset)
    
    # 3. Evaluate RAG
    k_neighbors = 5
    preds, truth = retrieve_and_predict(index, db_targets, query_dataset, k=k_neighbors)
    
    # 4. Compute Metrics
    mse = np.mean((preds - truth) ** 2)
    print(f"\n📊 RAG RESULTS (k={k_neighbors})")
    print("-" * 30)
    print(f"MSE: {mse:.6f}")
    
    # Compare with a global mean baseline (dumbest predictor)
    global_mean = np.mean(db_targets, axis=0)
    baseline_preds = np.tile(global_mean, (len(truth), 1))
    baseline_mse = np.mean((baseline_preds - truth) ** 2)
    print(f"Global Mean Baseline MSE: {baseline_mse:.6f}")
    
    if mse < baseline_mse:
        print("✅ RAG is working! (Better than random guessing)")
    else:
        print("❌ RAG needs improvement (fingerprints might not be specific enough)")
        
    # 5. Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(truth[:, 0], preds[:, 0], alpha=0.6, label='Gamma')
    plt.plot([0, 2*np.pi], [0, 2*np.pi], 'r--')
    plt.title(f"Gamma Prediction (MSE: {np.mean((preds[:,0]-truth[:,0])**2):.4f})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    
    plt.subplot(1, 2, 2)
    plt.scatter(truth[:, 1], preds[:, 1], alpha=0.6, color='orange', label='Beta')
    plt.plot([0, np.pi], [0, np.pi], 'r--')
    plt.title(f"Beta Prediction (MSE: {np.mean((preds[:,1]-truth[:,1])**2):.4f})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig('rag_prototype_results.png')
    print("📸 Saved plot to rag_prototype_results.png")

if __name__ == "__main__":
    run_experiment()

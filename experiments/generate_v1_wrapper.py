from generate_physics_proxy import generate_proxy_dataset
import json
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

# We need to reimplement heuristic features because the import might fail 
# if it depends on benchmark_encoding which has circular deps.
# Or simpler: we just save the graph and targets, and let benchmark_v2 compute features.

def serialize_graph(data, sample_id):
    """Convert PyG Data/NetworkX to the JSON format expected by V2 Benchmark"""
    G = data.G
    y = data.y.numpy()[0] # [gamma, beta]
    
    return {
        'id': sample_id,
        'adjacency_matrix': nx.to_numpy_array(G).tolist(),
        'n_nodes': G.number_of_nodes(),
        'optimal_gamma': [float(y[0])],
        'optimal_beta': [float(y[1])],
        'graph_type': 'PROXY_V1',
        'rwpe': None, # V1 doesn't have this precomputed
        'node_features': [[d] for _, d in G.degree()] # Minimal placeholder
    }

print("🚀 Generating V1 Proxy Dataset (500 samples)...")
raw_data = generate_proxy_dataset(num_graphs=500, min_nodes=10, max_nodes=16)

json_dataset = []
for i, data in enumerate(tqdm(raw_data)):
    json_dataset.append(serialize_graph(data, i))

out_path = "Dataset/qaoa_proxy_v1_500.json"
with open(out_path, 'w') as f:
    json.dump(json_dataset, f)

print(f"✅ V1 Dataset saved to {out_path}")

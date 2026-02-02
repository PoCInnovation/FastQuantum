#!/usr/bin/env python3
"""
Extract real metrics from the FastQuantum model and datasets.
Outputs a JSON file that the frontend can consume.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

def load_model_checkpoint(model_path: str) -> dict:
    """Load the saved model checkpoint and extract metrics."""
    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint

def load_dataset_stats(dataset_path: str) -> dict:
    """Load dataset and compute statistics."""
    if not os.path.exists(dataset_path):
        return None

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    n_nodes = [s['n_nodes'] for s in dataset]
    n_edges = [s['n_edges'] for s in dataset]

    # Compute feature statistics from first few samples
    all_features = []
    for sample in dataset[:100]:  # Sample subset for speed
        features = np.array(sample['node_features'])
        all_features.append(features)

    # Feature-wise statistics
    all_features_flat = np.vstack(all_features)
    feature_means = np.mean(all_features_flat, axis=0).tolist()
    feature_stds = np.std(all_features_flat, axis=0).tolist()

    # Gamma/Beta statistics
    gammas = [s['optimal_gamma'][0] for s in dataset]
    betas = [s['optimal_beta'][0] for s in dataset]

    return {
        'n_samples': len(dataset),
        'nodes': {
            'min': int(np.min(n_nodes)),
            'max': int(np.max(n_nodes)),
            'mean': float(np.mean(n_nodes)),
            'std': float(np.std(n_nodes))
        },
        'edges': {
            'min': int(np.min(n_edges)),
            'max': int(np.max(n_edges)),
            'mean': float(np.mean(n_edges)),
            'std': float(np.std(n_edges))
        },
        'features': {
            'means': feature_means,
            'stds': feature_stds
        },
        'gamma': {
            'min': float(np.min(gammas)),
            'max': float(np.max(gammas)),
            'mean': float(np.mean(gammas)),
            'std': float(np.std(gammas))
        },
        'beta': {
            'min': float(np.min(betas)),
            'max': float(np.max(betas)),
            'mean': float(np.mean(betas)),
            'std': float(np.std(betas))
        }
    }

def count_model_parameters(checkpoint: dict) -> int:
    """Count total trainable parameters from state dict."""
    if 'model_state_dict' not in checkpoint:
        return 0

    total = 0
    for key, tensor in checkpoint['model_state_dict'].items():
        total += tensor.numel()
    return total

def main():
    base_path = Path(__file__).parent.parent.parent

    # Paths
    model_path = base_path / 'best_qaoa_gat_model.pt'
    train_path = base_path / 'Dataset' / 'qaoa_train_dataset.json'
    val_path = base_path / 'Dataset' / 'qaoa_val_dataset.json'
    output_path = Path(__file__).parent.parent / 'public' / 'metrics.json'

    print("Extracting metrics from FastQuantum...")

    metrics = {
        'model': None,
        'training': None,
        'validation': None,
        'features': [
            {
                'name': 'Degree',
                'key': 'degree',
                'description': 'Nombre de connexions directes du noeud. Un noeud avec un degre eleve est tres connecte.',
                'formula': 'deg(v) = |N(v)|',
                'interpretation': 'Plus le degre est eleve, plus le noeud a d\'influence locale sur le graphe.'
            },
            {
                'name': 'Degree Centrality',
                'key': 'degree_centrality',
                'description': 'Degre normalise par le nombre maximum de connexions possibles.',
                'formula': 'C_D(v) = deg(v) / (n-1)',
                'interpretation': 'Valeur entre 0 et 1. Permet de comparer des graphes de tailles differentes.'
            },
            {
                'name': 'Clustering Coefficient',
                'key': 'clustering',
                'description': 'Mesure a quel point les voisins d\'un noeud sont connectes entre eux.',
                'formula': 'C(v) = 2*e_v / (k_v * (k_v-1))',
                'interpretation': 'Proche de 1 = les voisins forment une clique. Proche de 0 = voisins isoles.'
            },
            {
                'name': 'Betweenness Centrality',
                'key': 'betweenness',
                'description': 'Proportion de plus courts chemins passant par ce noeud.',
                'formula': 'C_B(v) = sum(sigma_st(v) / sigma_st)',
                'interpretation': 'Noeud "pont" important pour la communication dans le reseau.'
            },
            {
                'name': 'Closeness Centrality',
                'key': 'closeness',
                'description': 'Inverse de la distance moyenne aux autres noeuds.',
                'formula': 'C_C(v) = (n-1) / sum(d(v,u))',
                'interpretation': 'Noeud central qui peut atteindre tous les autres rapidement.'
            },
            {
                'name': 'PageRank',
                'key': 'pagerank',
                'description': 'Importance basee sur les connexions entrantes ponderees (algorithme Google).',
                'formula': 'PR(v) = (1-d)/n + d * sum(PR(u)/L(u))',
                'interpretation': 'Un noeud est important si des noeuds importants pointent vers lui.'
            },
            {
                'name': 'Eigenvector Centrality',
                'key': 'eigenvector',
                'description': 'Importance basee sur l\'importance des voisins.',
                'formula': 'x_v = (1/lambda) * sum(x_u)',
                'interpretation': 'Mesure l\'influence globale dans le reseau.'
            }
        ]
    }

    # Load model checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = load_model_checkpoint(str(model_path))
    if checkpoint:
        metrics['model'] = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': float(checkpoint.get('val_loss', 0)),
            'gamma_mae': float(checkpoint.get('gamma_mae', 0)),
            'beta_mae': float(checkpoint.get('beta_mae', 0)),
            'p_layers': checkpoint.get('p_layers', 1),
            'model_type': checkpoint.get('model_type', 'GAT'),
            'total_parameters': count_model_parameters(checkpoint)
        }
        print(f"  - Epoch: {metrics['model']['epoch']}")
        print(f"  - Val Loss: {metrics['model']['val_loss']:.6f}")
        print(f"  - Gamma MAE: {metrics['model']['gamma_mae']:.6f}")
        print(f"  - Beta MAE: {metrics['model']['beta_mae']:.6f}")
        print(f"  - Parameters: {metrics['model']['total_parameters']:,}")
    else:
        print("  - Model not found!")

    # Load training dataset stats
    print(f"\nLoading training dataset from {train_path}...")
    train_stats = load_dataset_stats(str(train_path))
    if train_stats:
        metrics['training'] = train_stats
        print(f"  - Samples: {train_stats['n_samples']}")
        print(f"  - Nodes: {train_stats['nodes']['min']}-{train_stats['nodes']['max']} (avg: {train_stats['nodes']['mean']:.1f})")
        print(f"  - Edges: {train_stats['edges']['min']}-{train_stats['edges']['max']} (avg: {train_stats['edges']['mean']:.1f})")
    else:
        print("  - Training dataset not found!")

    # Load validation dataset stats
    print(f"\nLoading validation dataset from {val_path}...")
    val_stats = load_dataset_stats(str(val_path))
    if val_stats:
        metrics['validation'] = val_stats
        print(f"  - Samples: {val_stats['n_samples']}")
        print(f"  - Nodes: {val_stats['nodes']['min']}-{val_stats['nodes']['max']} (avg: {val_stats['nodes']['mean']:.1f})")
    else:
        print("  - Validation dataset not found!")

    # Save metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_path}")

if __name__ == '__main__':
    main()

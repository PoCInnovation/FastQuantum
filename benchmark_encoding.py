"""
Benchmark Script: Comparing Positional Encoding Strategies for QAOA Parameter Prediction
=========================================================================================

This script compares three approaches:
1. Baseline: Heuristic features only (7 dims)
2. LPE: Heuristics + Laplacian Positional Encodings (k=4)
3. RWPE: Heuristics + Random Walk Positional Encodings (walk_length=16)

Author: FastQuantum Project
Date: 2026-01-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}\n")


# ============================================================================
# 1. HELPERS
# ============================================================================

def compute_heuristic_features(G):
    """
    Compute 7-dimensional heuristic features for each node.
    Features: [degree_norm, clustering_coef, betweenness_cent, closeness_cent, 
               pagerank, eigenvector_cent, core_number_norm]
    """
    n = G.number_of_nodes()
    
    # Degree (normalized)
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    degree_norm = np.array([degrees[i] / max_deg for i in range(n)])
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    clustering_coef = np.array([clustering[i] for i in range(n)])
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    betweenness_cent = np.array([betweenness[i] for i in range(n)])
    
    # Closeness centrality
    closeness = nx.closeness_centrality(G)
    closeness_cent = np.array([closeness[i] for i in range(n)])
    
    # PageRank
    pagerank = nx.pagerank(G)
    pagerank_vals = np.array([pagerank[i] for i in range(n)])
    
    # Eigenvector centrality (with fallback)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        eigenvector_cent = np.array([eigenvector[i] for i in range(n)])
    except:
        eigenvector_cent = np.zeros(n)
    
    # Core number (normalized)
    core_number = nx.core_number(G)
    max_core = max(core_number.values()) if core_number else 1
    core_number_norm = np.array([core_number[i] / max_core for i in range(n)])
    
    # Stack features
    features = np.stack([
        degree_norm,
        clustering_coef,
        betweenness_cent,
        closeness_cent,
        pagerank_vals,
        eigenvector_cent,
        core_number_norm
    ], axis=1)
    
    return features


# ============================================================================
# 2. MODEL DEFINITION
# ============================================================================

class GATModel(nn.Module):
    """
    GAT-based model for QAOA parameter prediction.
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, heads=4):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                  edge_dim=1, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                      heads=heads, edge_dim=1, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Last layer (no concat)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                  heads=heads, edge_dim=1, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output: (gamma, beta)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP
        out = self.mlp(x)
        return out


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device=DEVICE):
    """Full training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
    
    return train_losses, val_losses


# ============================================================================
# 4. BENCHMARK EXECUTION
# ============================================================================

def run_benchmark():
    """Main benchmark function."""
    print("=" * 80)
    print("BENCHMARK: Positional Encoding Strategies for QAOA Parameter Prediction")
    print("=" * 80)
    print()
    
    # Generate dataset
    # Dynamic import to avoid circular dependency or missing file issues at startup
    from generate_physics_proxy import generate_proxy_dataset
    
    print("📊 Generating Physics-Proxy dataset (1000 graphs, 40-60 nodes each)...")
    # Generate raw data with physics targets
    raw_dataset = generate_proxy_dataset(num_graphs=1000, min_nodes=40, max_nodes=60)
    
    # Compute heuristic features for all graphs
    print("⚙️  Computing heuristic features for all graphs...")
    dataset = []
    for data in tqdm(raw_dataset, desc="Feature Engineering"):
        if hasattr(data, 'G'):
            x = compute_heuristic_features(data.G)
            data.x = torch.tensor(x, dtype=torch.float)
            
            # Edge weights need to be added if not present (proxy generator adds index but not attrs)
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                 data.edge_attr = torch.rand(data.edge_index.size(1), 1)

            dataset.append(data)
        else:
            print("Warning: Data object missing Graph object 'G'")
    
    # Split dataset (800 Train, 100 Val, 100 Test)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:900]
    test_dataset = dataset[900:]
    
    print(f"✅ Dataset generated: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test\n")
    
    # Results storage
    results = {}
    
    # Common training params
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # ========================================================================
    # VARIANT 1: BASELINE (Heuristics only)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔵 VARIANT 1: BASELINE (Heuristic Features Only - 7 dims)")
    print("=" * 80)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    in_channels = train_dataset[0].x.size(1)
    print(f"Input dimension: {in_channels}")
    
    model_baseline = GATModel(in_channels=in_channels)
    start_time = time.time()
    train_losses_baseline, val_losses_baseline = train_model(model_baseline, train_loader, val_loader, num_epochs=EPOCHS)
    elapsed_time_baseline = time.time() - start_time
    
    results['Baseline'] = {
        'train_losses': train_losses_baseline,
        'val_losses': val_losses_baseline,
        'final_val_loss': val_losses_baseline[-1],
        'time': elapsed_time_baseline
    }
    
    print(f"\n✅ Baseline completed in {elapsed_time_baseline:.2f}s")
    print(f"   Final Validation Loss: {val_losses_baseline[-1]:.4f}\n")
    
    # ========================================================================
    # VARIANT 2: LPE (Laplacian Positional Encodings)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🟢 VARIANT 2: LPE (Heuristics + Laplacian PE, k=4)")
    print("=" * 80)
    
    # Apply LPE transform with error handling
    lpe_transform = AddLaplacianEigenvectorPE(k=4, attr_name='laplacian_pe')
    train_dataset_lpe = []
    val_dataset_lpe = []
    
    print("Applying LPE transform to train set...")
    for data in tqdm(train_dataset):
        try:
            data_transformed = lpe_transform(data.clone())
            # Concatenate PE to features
            data_transformed.x = torch.cat([data_transformed.x, data_transformed.laplacian_pe], dim=1)
            train_dataset_lpe.append(data_transformed)
        except Exception as e:
            # Fallback: pad with zeros if LPE fails
            data_padded = data.clone()
            data_padded.x = torch.cat([data_padded.x, torch.zeros(data.x.size(0), 4)], dim=1)
            train_dataset_lpe.append(data_padded)
    
    print("Applying LPE transform to val set...")
    for data in tqdm(val_dataset):
        try:
            data_transformed = lpe_transform(data.clone())
            data_transformed.x = torch.cat([data_transformed.x, data_transformed.laplacian_pe], dim=1)
            val_dataset_lpe.append(data_transformed)
        except Exception as e:
            data_padded = data.clone()
            data_padded.x = torch.cat([data_padded.x, torch.zeros(data.x.size(0), 4)], dim=1)
            val_dataset_lpe.append(data_padded)
    
    train_loader_lpe = DataLoader(train_dataset_lpe, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_lpe = DataLoader(val_dataset_lpe, batch_size=BATCH_SIZE, shuffle=False)
    
    in_channels_lpe = train_dataset_lpe[0].x.size(1)
    print(f"Input dimension: {in_channels_lpe}")
    
    model_lpe = GATModel(in_channels=in_channels_lpe)
    start_time = time.time()
    train_losses_lpe, val_losses_lpe = train_model(model_lpe, train_loader_lpe, val_loader_lpe, num_epochs=EPOCHS)
    elapsed_time_lpe = time.time() - start_time
    
    results['LPE'] = {
        'train_losses': train_losses_lpe,
        'val_losses': val_losses_lpe,
        'final_val_loss': val_losses_lpe[-1],
        'time': elapsed_time_lpe
    }
    
    print(f"\n✅ LPE completed in {elapsed_time_lpe:.2f}s")
    print(f"   Final Validation Loss: {val_losses_lpe[-1]:.4f}\n")
    
    # ========================================================================
    # VARIANT 3: RWPE (Random Walk Positional Encodings)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🟣 VARIANT 3: RWPE (Heuristics + Random Walk PE, walk_length=16)")
    print("=" * 80)
    
    # Apply RWPE transform
    rwpe_transform = AddRandomWalkPE(walk_length=16, attr_name='random_walk_pe')
    train_dataset_rwpe = []
    val_dataset_rwpe = []
    
    print("Applying RWPE transform to train set...")
    for data in tqdm(train_dataset):
        data_transformed = rwpe_transform(data.clone())
        data_transformed.x = torch.cat([data_transformed.x, data_transformed.random_walk_pe], dim=1)
        train_dataset_rwpe.append(data_transformed)
    
    print("Applying RWPE transform to val set...")
    for data in tqdm(val_dataset):
        data_transformed = rwpe_transform(data.clone())
        data_transformed.x = torch.cat([data_transformed.x, data_transformed.random_walk_pe], dim=1)
        val_dataset_rwpe.append(data_transformed)
    
    train_loader_rwpe = DataLoader(train_dataset_rwpe, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_rwpe = DataLoader(val_dataset_rwpe, batch_size=BATCH_SIZE, shuffle=False)
    
    in_channels_rwpe = train_dataset_rwpe[0].x.size(1)
    print(f"Input dimension: {in_channels_rwpe}")
    
    model_rwpe = GATModel(in_channels=in_channels_rwpe)
    start_time = time.time()
    train_losses_rwpe, val_losses_rwpe = train_model(model_rwpe, train_loader_rwpe, val_loader_rwpe, num_epochs=EPOCHS)
    elapsed_time_rwpe = time.time() - start_time
    
    results['RWPE'] = {
        'train_losses': train_losses_rwpe,
        'val_losses': val_losses_rwpe,
        'final_val_loss': val_losses_rwpe[-1],
        'time': elapsed_time_rwpe
    }
    
    print(f"\n✅ RWPE completed in {elapsed_time_rwpe:.2f}s")
    print(f"   Final Validation Loss: {val_losses_rwpe[-1]:.4f}\n")
    
    # ========================================================================
    # RESULTS VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    # Plot validation losses
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['Baseline']['val_losses'], label='Baseline (Heuristics)', linewidth=2, color='#3498db')
    plt.plot(results['LPE']['val_losses'], label='LPE (k=4)', linewidth=2, color='#2ecc71')
    plt.plot(results['RWPE']['val_losses'], label='RWPE (walk=16)', linewidth=2, color='#9b59b6')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontsize=12)
    plt.title('Validation Loss Comparison (Physics Proxy)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['Baseline']['train_losses'], label='Baseline (Heuristics)', linewidth=2, color='#3498db', alpha=0.7)
    plt.plot(results['LPE']['train_losses'], label='LPE (k=4)', linewidth=2, color='#2ecc71', alpha=0.7)
    plt.plot(results['RWPE']['train_losses'], label='RWPE (walk=16)', linewidth=2, color='#9b59b6', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (MSE)', fontsize=12)
    plt.title('Training Loss Comparison (Physics Proxy)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/nerzouille/Delivery/PoC/FastQuantum/benchmark_results_physics.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to: benchmark_results_physics.png")
    # plt.show()
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY (PHYSICS PROXY)")
    print("=" * 80)
    print(f"{'Variant':<20} {'Final Val Loss':<20} {'Training Time (s)':<20}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<20} {res['final_val_loss']:<20.6f} {res['time']:<20.2f}")
    print("=" * 80)
    
    # Determine winner
    winner = min(results.items(), key=lambda x: x[1]['final_val_loss'])
    print(f"\n🏆 WINNER: {winner[0]} with validation loss of {winner[1]['final_val_loss']:.6f}")
    if winner[0] != 'Baseline':
        print("🚀 HYPOTHESIS CONFIRMED: Positional Encodings outperform Baseline on Physics-Aware Data!")
    else:
        print("🤔 Baseline still wins. Consider checking proxy correlation or hyperparameters.")
    
    print("\n✅ Benchmark completed successfully!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_benchmark()

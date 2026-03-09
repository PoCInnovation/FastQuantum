"""
Benchmark: RAG vs GNN
=====================

Compares the "Memory-Based" (RAG) approach against the "Model-Based" (GNN) approach
for QAOA parameter prediction.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our new tools
from utils.graph_memory import GraphMemory
from generate_physics_proxy import generate_proxy_dataset
from benchmark_encoding import compute_heuristic_features

# ============================================================================
# 1. GNN MODEL DEFINITION (Baseline)
# ============================================================================
class GATBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(GATBaseline, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels*4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels*4, hidden_channels, heads=4, concat=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels*2, 64), # *2 for mean+max pool
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Gamma, Beta
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.mlp(x)

def train_gnn(model, train_loader, epochs=50, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    
    losses = []
    print(f"🧠 Training GNN Baseline ({epochs} epochs)...")
    for _ in tqdm(range(epochs)):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
    return model, losses

# ============================================================================
# 2. BENCHMARK EXECUTION
# ============================================================================
def run_benchmark():
    print("🚀 STARTING BENCHMARK: RAG (Memory) vs GNN (Model)")
    print("="*60)
    
    # 1. Generate Data
    # ----------------
    print("\n[1/4] Generating Dataset...")
    # 1200 Graphs: 1000 for Train/Memory, 200 for Test
    dataset = generate_proxy_dataset(num_graphs=1200, min_nodes=30, max_nodes=50)
    
    # Add features for GNN
    for data in dataset:
        data.x = torch.tensor(compute_heuristic_features(data.G), dtype=torch.float)
    
    train_data = dataset[:1000]
    test_data = dataset[1000:]
    
    # 2. Train/Fit Systems
    # --------------------
    print("\n[2/4] Training Systems...")
    
    # System A: RAG (Memory)
    start_rag = time.time()
    memory = GraphMemory(fingerprint_dim=32) # Using 32 to be safe with padding
    
    # Extract graphs and targets for memory
    train_graphs = [d.G for d in train_data]
    train_targets = [d.y.numpy().flatten() for d in train_data]
    
    memory.add_graphs(train_graphs, train_targets)
    time_rag_train = time.time() - start_rag
    print(f"✅ RAG Indexed 1000 graphs in {time_rag_train:.2f}s")
    
    # System B: GNN (Model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    gnn = GATBaseline(in_channels=7).to(device)
    start_gnn = time.time()
    gnn, _ = train_gnn(gnn, train_loader, epochs=30, device=device) # Reduced epochs for speed
    time_gnn_train = time.time() - start_gnn
    print(f"✅ GNN Trained in {time_gnn_train:.2f}s")

    # 3. Evaluation
    # 1/4
    print("\n[3/4] Evaluating on Test Set (200 graphs)...")
    
    rag_preds = []
    gnn_preds = []
    truth = []
    
    rag_times = []
    gnn_times = []
    
    gnn.eval()
    
    for data in tqdm(test_data):
        target = data.y.numpy().flatten()
        truth.append(target)
        
        # Test RAG
        t0 = time.time()
        pred_rag, _ = memory.retrieve(data.G, k=5)
        rag_times.append(time.time() - t0)
        rag_preds.append(pred_rag)
        
        # Test GNN
        t0 = time.time()
        with torch.no_grad():
            d_gpu = data.to(device)
            # Batch of 1 requires unsqueeze if dimensions are lost, but PyG handles it usually
            # data.batch needs to be set manually for single instance or use loader
            d_gpu.batch = torch.zeros(d_gpu.x.size(0), dtype=torch.long, device=device)
            pred_gnn = gnn(d_gpu).cpu().numpy().flatten()
        gnn_times.append(time.time() - t0)
        gnn_preds.append(pred_gnn)

    rag_preds = np.array(rag_preds)
    gnn_preds = np.array(gnn_preds)
    truth = np.array(truth)
    
    # 4. Results
    # ----------
    mse_rag = np.mean((rag_preds - truth)**2)
    mse_gnn = np.mean((gnn_preds - truth)**2)
    
    avg_time_rag = np.mean(rag_times) * 1000 # ms
    avg_time_gnn = np.mean(gnn_times) * 1000 # ms
    
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    print(f"{ 'Metric':<20} | { 'RAG (Memory)':<15} | { 'GNN (Model)':<15}")
    print("-" * 60)
    print(f"{ 'MSE (Loss)':<20} | {mse_rag:.6f}        | {mse_gnn:.6f}")
    print(f"{ 'Training Time':<20} | {time_rag_train:.2f}s           | {time_gnn_train:.2f}s")
    print(f"{ 'Inference (ms/g)':<20} | {avg_time_rag:.2f}ms          | {avg_time_gnn:.2f}ms")
    print("-" * 60)
    
    if mse_rag < mse_gnn:
        print("\n🎉 WINNER: RAG (Memory-Based)!")
        print("Reason: Topological similarity is a stronger predictor than generalized physics for this dataset.")
    else:
        print("\n🎉 WINNER: GNN (Model-Based)!")
        print("Reason: The model successfully learned the underlying physical laws.")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(truth[:, 0], rag_preds[:, 0], alpha=0.5, label='RAG')
    plt.scatter(truth[:, 0], gnn_preds[:, 0], alpha=0.5, label='GNN', marker='x')
    plt.plot([0, 6], [0, 6], 'k--')
    plt.title("Gamma Prediction")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(truth[:, 1], rag_preds[:, 1], alpha=0.5, label='RAG')
    plt.scatter(truth[:, 1], gnn_preds[:, 1], alpha=0.5, label='GNN', marker='x')
    plt.plot([0, 3], [0, 3], 'k--')
    plt.title("Beta Prediction")
    plt.legend()
    
    plt.savefig('benchmark_rag_vs_gnn.png')
    print("📸 Plot saved to benchmark_rag_vs_gnn.png")

if __name__ == "__main__":
    run_benchmark()

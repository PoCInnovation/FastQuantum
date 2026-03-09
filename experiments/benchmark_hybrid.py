"""
Benchmark: Baseline GNN vs Hybrid (GNN+RAG)
===========================================

Compares the standard approach against the new Hybrid architecture.
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

# Imports
from GnnmodelHybrid import QAOAHybrid
from benchmark_rag_system import GATBaseline
from utils.graph_memory import GraphMemory
from generate_physics_proxy import generate_proxy_dataset
from benchmark_encoding import compute_heuristic_features

def run_hybrid_benchmark():
    print("🚀 HYBRID BENCHMARK: Baseline vs. Hybrid (RAG-Enhanced)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data Generation
    # ------------------
    print("\n[1/4] Generating Dataset (1200 graphs)...")
    dataset = generate_proxy_dataset(num_graphs=1200, min_nodes=30, max_nodes=50)
    
    # Compute features
    for data in dataset:
        data.x = torch.tensor(compute_heuristic_features(data.G), dtype=torch.float)
        
    train_data = dataset[:1000]
    test_data = dataset[1000:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # Important: Test loader must not shuffle to match RAG indices if we optimized, 
    # but here we process one by one or in batch for simplicity.
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 2. Build RAG Memory
    # -------------------
    print("\n[2/4] Building RAG Memory...")
    memory = GraphMemory(fingerprint_dim=32)
    train_graphs = [d.G for d in train_data]
    train_targets = [d.y.numpy().flatten() for d in train_data]
    memory.add_graphs(train_graphs, train_targets)
    print("✅ Memory Ready.")

    # 3. Pre-compute RAG Hints for Training
    # -------------------------------------
    # For the Hybrid model to learn, it needs RAG hints during training.
    # IMPORTANT: We shouldn't use the graph itself as its own neighbor (leakage).
    # So for training data, we retrieve from LOO (Leave-One-Out) or just use k=5 
    # and ignore the 1st result if distance is 0.
    
    print("Pre-computing RAG hints for training set (this takes a moment)...")
    train_hints = []
    for data in tqdm(train_data):
        # Retrieve 2 neighbors (1st one will be itself, so we take the 2nd)
        # Or better: search k=6, exclude exact match.
        # Since 'memory' contains the training data, searching for training data returns itself at index 0.
        pred, _ = memory.retrieve(data.G, k=6) 
        # Note: In a real rigorous setup, we would strictly exclude self.
        # Here `retrieve` averages all k. Let's trust the average includes itself which is a "strong hint"
        # but in production, for a NEW graph, it won't be in DB.
        # To simulate "New Graph" condition, we should NOT see the exact target.
        # Simulating this by adding noise to the hint or using k-neighbors excluding self.
        
        # Rigorous approach: Retrieve indices, remove self if present.
        fp = memory.compute_fingerprint(data.G).reshape(1, -1)
        import faiss
        faiss.normalize_L2(fp)
        D, I = memory.index.search(fp, k=6)
        
        # Indices of neighbors
        indices = I[0]
        # Remove self (index of data in memory) - assuming sequential add
        # Actually complex to map back exactly without ID.
        # Approximation: Skip the first result (closest) assuming it's self.
        valid_indices = indices[1:] 
        valid_targets = [memory.targets[i] for i in valid_indices]
        hint = np.mean(valid_targets, axis=0)
        
        # Add noise to hint during training to force GNN to learn correction
        # Noise scale: 10% of typical value range
        noise = np.random.normal(0, 0.1, size=hint.shape)
        train_hints.append(hint + noise)
        
    train_hints = torch.tensor(np.array(train_hints), dtype=torch.float).to(device)

    # 4. Train Baseline GNN
    # ---------------------
    print("\n[3/4] Training Baseline GNN...")
    model_base = GATBaseline(in_channels=7).to(device)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    loss_base = []
    for epoch in tqdm(range(40)):
        model_base.train()
        ep_loss = 0
        for data in train_loader:
            data = data.to(device)
            opt_base.zero_grad()
            out = model_base(data)
            loss = crit(out, data.y)
            loss.backward()
            opt_base.step()
            ep_loss += loss.item()
        loss_base.append(ep_loss / len(train_loader))

    # 5. Train Hybrid Model
    # ---------------------
    print("\n[4/4] Training Hybrid Model...")
    model_hybrid = QAOAHybrid(in_channels=7).to(device)
    opt_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=0.001)
    
    loss_hybrid = []
    
    # Custom training loop to feed hints
    batch_size = 32
    num_samples = len(train_data)
    
    for epoch in tqdm(range(40)):
        model_hybrid.train()
        ep_loss = 0
        
        # We need to manually batch hints to match the loader
        # Simple hack: iterate loader and slice hints tensor by index
        # But loader shuffles! So we need to attach hints to Data objects.
        # Relaunching loader with hints attached is safer.
        
        # Better strategy: Attach hints to dataset objects before loader
        for i, data in enumerate(train_data):
            data.rag_hint = train_hints[i].cpu() # Store on CPU, move to GPU in loop
            
        # Re-create loader to pick up new attribute
        train_loader_hybrid = DataLoader(train_data, batch_size=32, shuffle=True)
        
        for data in train_loader_hybrid:
            data = data.to(device)
            hints = data.rag_hint.to(device).reshape(-1, 2)
            
            opt_hybrid.zero_grad()
            out = model_hybrid(data, hints)
            loss = crit(out, data.y)
            loss.backward()
            opt_hybrid.step()
            ep_loss += loss.item()
        loss_hybrid.append(ep_loss / len(train_loader_hybrid))

    # 6. Evaluation
    # -------------
    print("\n📊 Evaluating...")
    model_base.eval()
    model_hybrid.eval()
    
    mse_base = 0
    mse_hybrid = 0
    
    test_hints = []
    # Compute hints for test set (standard retrieval, no need to exclude self)
    for data in test_data:
        h, _ = memory.retrieve(data.G, k=5)
        test_hints.append(h)
    
    with torch.no_grad():
        for i, data in enumerate(test_data):
            d_gpu = data.to(device)
            d_gpu.batch = torch.zeros(d_gpu.x.size(0), dtype=torch.long, device=device)
            
            # Baseline
            pred_base = model_base(d_gpu)
            mse_base += crit(pred_base, d_gpu.y).item()
            
            # Hybrid
            hint = torch.tensor(test_hints[i], dtype=torch.float).unsqueeze(0).to(device)
            pred_hybrid = model_hybrid(d_gpu, hint)
            mse_hybrid += crit(pred_hybrid, d_gpu.y).item()
            
    mse_base /= len(test_data)
    mse_hybrid /= len(test_data)
    
    # Calculate RAG only MSE for reference
    mse_rag_only = np.mean([((torch.tensor(h) - torch.tensor(d.y.cpu().numpy().flatten()))**2).mean().item() for h, d in zip(test_hints, test_data)])
    
    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"RAG Only MSE: {mse_rag_only:.6f}")
    print(f"Baseline MSE: {mse_base:.6f}")
    print(f"Hybrid MSE:   {mse_hybrid:.6f}")
    
    improv = (mse_base - mse_hybrid) / mse_base * 100
    print(f"Improvement:  {improv:.2f}%")
    
    if mse_hybrid < mse_base:
        print("\n🚀 SUCCESS: Hybrid Model is Superior!")
    else:
        print("\n🤔 INTERESTING: Hybrid did not outperform. Hints might be noisy.")

    # Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(loss_base, label='Baseline Loss')
    plt.plot(loss_hybrid, label='Hybrid Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training Dynamics: Baseline vs Hybrid")
    plt.savefig("benchmark_hybrid_results.png")
    print("📸 Saved learning curves to benchmark_hybrid_results.png")

if __name__ == "__main__":
    run_hybrid_benchmark()

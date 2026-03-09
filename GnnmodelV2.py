import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class FastQuantumV2(nn.Module):
    """
    FastQuantum V2: Hybrid Model
    Combines:
    1. Physics-Aware GNN (Heuristics + RWPE)
    2. Categorical ID Embeddings (for fine-tuning specific circuits)
    """
    def __init__(self, in_channels, num_known_circuits=1000, embedding_dim=16, 
                 hidden_channels=64, num_layers=3, heads=4):
        super(FastQuantumV2, self).__init__()
        
        # --- PART 1: PHYSICS ENGINE (GNN) ---
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
        
        # Last GNN layer
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                  heads=heads, edge_dim=1, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # --- PART 2: IDENTITY ENGINE (Embeddings) ---
        self.embedding_dim = embedding_dim
        self.circuit_embedding = nn.Embedding(num_embeddings=num_known_circuits, 
                                              embedding_dim=embedding_dim)
        
        # --- PART 3: FUSION & HEAD ---
        # Input size = GNN Output (Mean+Max Pool) + ID Embedding
        gnn_out_dim = hidden_channels * 2 
        combined_dim = gnn_out_dim + embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output: (gamma, beta)
        )

    def forward(self, data, circuit_ids):
        """
        data: PyG Data object (x, edge_index, etc.)
        circuit_ids: Tensor of shape [batch_size] containing circuit IDs
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Physics Path (GNN)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # Global Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_feat = torch.cat([x_mean, x_max], dim=1)
        
        # 2. Identity Path (Embedding)
        # circuit_ids should be a tensor of IDs corresponding to each graph in the batch
        id_feat = self.circuit_embedding(circuit_ids)
        
        # 3. Fusion
        combined = torch.cat([graph_feat, id_feat], dim=1)
        
        # Prediction
        out = self.mlp(combined)
        return out

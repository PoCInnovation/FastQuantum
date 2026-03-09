import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class QAOAHybrid(nn.Module):
    """
    Hybrid Model: GNN Physics + RAG Memory
    
    Architecture:
    1. Physics Branch (GAT): Extracts topological features from the graph.
    2. Memory Branch (MLP): Processes the 'hint' provided by the RAG system.
    3. Fusion Layer: Combines both signals to predict optimal parameters.
    """
    def __init__(self, in_channels, hidden_channels=64, rag_input_dim=2):
        super(QAOAHybrid, self).__init__()
        
        # --- Physics Branch (GAT) ---
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels*4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels*4, hidden_channels, heads=4, concat=False)
        

        # --- Residual Head ---
        # The GNN predicts a DELTA (correction) to the RAG hint
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64), # *2 for mean+max pool
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: Delta Gamma, Delta Beta
        )

    def forward(self, data, rag_hint):
        """
        Args:
            data: PyG Data object (graph)
            rag_hint: Tensor of shape (batch_size, 2) containing RAG predictions
        """
        # 1. Physics Path (GNN)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Global Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1) # Shape: [Batch, Hidden*2]
        
        # 2. Delta Prediction
        delta = self.delta_mlp(x_graph)
        
        # 3. Final Prediction = RAG Hint + Delta
        out = rag_hint + delta
        
        return out

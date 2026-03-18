import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class QAOAPredictorGCN(nn.Module):
    """
    Graph Convolutional Network (GCN) baseline for predicting QAOA parameters.
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, p_layers=1, dropout=0.3):
        super(QAOAPredictorGCN, self).__init__()
        
        self.num_layers = num_layers
        self.p_layers = p_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # MLP head
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_gamma = nn.Linear(64, p_layers)
        self.fc_beta = nn.Linear(64, p_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            if i > 0 and x.shape[1] == x_new.shape[1]:
                x = x + x_new
            else:
                x = x_new
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        gamma = self.fc_gamma(x)
        beta = self.fc_beta(x)
        
        output = torch.cat([gamma, beta], dim=1)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class QAOAPredictorGAT(nn.Module):
    """
    Graph Attention Network (GAT) for predicting QAOA parameters.

    This model utilizes attention mechanisms to weigh the importance of neighbors,
    explicitly incorporating edge weights into the attention scores.
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, p_layers=1,
                 attention_heads=8, dropout=0.3, use_edge_features=True, use_input_bn=True):
        super(QAOAPredictorGAT, self).__init__()

        self.num_layers = num_layers
        self.p_layers = p_layers
        self.attention_heads = attention_heads
        self.use_edge_features = use_edge_features
        self.use_input_bn = use_input_bn

        edge_dim = 1 if use_edge_features else None

        # GAT layers with multi-head attention
        self.convs = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.convs.append(GATConv(
            input_dim,
            hidden_dim // attention_heads,
            heads=attention_heads,
            dropout=dropout,
            concat=True,
            edge_dim=edge_dim
        ))

        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim,
                hidden_dim // attention_heads,
                heads=attention_heads,
                dropout=dropout,
                concat=True,
                edge_dim=edge_dim
            ))

        # Last layer: hidden_dim -> hidden_dim (average heads instead of concat)
        self.convs.append(GATConv(
            hidden_dim,
            hidden_dim,
            heads=attention_heads,
            dropout=dropout,
            concat=False,  # Average heads for final layer
            edge_dim=edge_dim
        ))

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Graph-level pooling (combine mean and max for richer representation)
        self.use_dual_pooling = True

        # MLP head for parameter prediction
        if self.use_dual_pooling:
            self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 because of mean+max pooling
        else:
            self.fc1 = nn.Linear(hidden_dim, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc_gamma = nn.Linear(64, p_layers)
        self.fc_beta = nn.Linear(64, p_layers)

        self.dropout = nn.Dropout(dropout)

        # Input Batch Normalization (optional)
        if use_input_bn:
            self.input_bn = nn.BatchNorm1d(input_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.use_input_bn:
            x = self.input_bn(x)

        edge_attr = getattr(data, 'edge_attr', None) if self.use_edge_features else None

        # GAT layers with residual connections
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr=edge_attr)

            # Batch normalization
            x_new = self.batch_norms[i](x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection (skip connection)
            if i > 0 and x.shape[1] == x_new.shape[1]:
                x = x + x_new
            else:
                x = x_new

        if self.use_dual_pooling:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x = global_mean_pool(x, batch)

        # MLP head
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)

        # Predict gamma and beta
        gamma = self.fc_gamma(x)
        beta = self.fc_beta(x)

        return torch.cat([gamma, beta], dim=1)

    def get_attention_weights(self, data):
        device = next(self.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device) if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        attention_weights = []

        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
                x_new, (edge_idx, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attention_weights.append((edge_idx.cpu(), alpha.cpu()))
                x_new = self.batch_norms[i](x_new)
                x_new = F.elu(x_new)
                if i > 0 and x.shape[1] == x_new.shape[1]:
                    x = x + x_new
                else:
                    x = x_new

        return attention_weights

"""
GNN Encoder for FastQuantum library.

Architecture: Stack of GAT layers with residual connections.
Outputs per-node embeddings (local) and a graph-level embedding (global via pooling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    GNN Encoder based on GAT.

    Input:  Graph (x, edge_index)
    Output:
        - node_embeddings: [n_nodes, hidden_dim] (local embeddings per node)
        - graph_embedding: [batch_size, hidden_dim] (global embedding via pooling)
    """

    def __init__(
        self,
        input_dim=23,
        hidden_dim=1024,
        num_layers=5,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x_new = gat(x, edge_index)
            x_new = norm(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)

            if i > 0:
                x = x + x_new
            else:
                x = x_new

        node_embeddings = x
        graph_embedding = global_mean_pool(node_embeddings, batch)

        return node_embeddings, graph_embedding

"""
Encodeur GNN pour le prototype v1

Architecture simplifiée:
- GAT layers pour encoder la structure du graphe
- Produit: node_embeddings [batch, n_nodes, hidden_dim]
- Produit: graph_embedding [batch, hidden_dim] via pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool


class GNNEncoder(nn.Module):
    """
    Encodeur GNN basé sur GAT.

    Entrée: Graphe (x, edge_index)
    Sortie:
        - node_embeddings: [n_nodes, hidden_dim] (embeddings locaux)
        - graph_embedding: [batch_size, hidden_dim] (embedding global via pooling)
    """

    def __init__(
        self,
        input_dim=7,           # Dimension des features de nœuds
        hidden_dim=128,        # Dimension des embeddings (128 comme dans le diagramme)
        num_layers=4,          # Nombre de couches GAT
        num_heads=4,           # Nombre de têtes d'attention
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Couches GAT
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Première couche: input_dim -> hidden_dim
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

        # Couches intermédiaires: hidden_dim -> hidden_dim
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
        """
        Args:
            x: [n_nodes, input_dim] - features des nœuds
            edge_index: [2, n_edges] - arêtes
            batch: [n_nodes] - assignation des nœuds aux graphes du batch

        Returns:
            node_embeddings: [n_nodes, hidden_dim]
            graph_embedding: [batch_size, hidden_dim]
        """
        # Gérer le cas sans batch (un seul graphe)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Passer à travers les couches GAT
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x_new = gat(x, edge_index)
            x_new = norm(x_new)
            x_new = F.elu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection après la première couche
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        node_embeddings = x  # [n_nodes, hidden_dim]

        # Pooling global pour l'embedding du graphe
        # Utilise mean pooling (comme dans le diagramme)
        graph_embedding = global_mean_pool(node_embeddings, batch)  # [batch_size, hidden_dim]

        return node_embeddings, graph_embedding


def test_encoder():
    """Test de l'encodeur"""
    print("=== Test GNNEncoder ===\n")

    # Paramètres
    n_nodes = 6
    input_dim = 7
    hidden_dim = 128

    # Données de test
    x = torch.randn(n_nodes, input_dim)
    # Graphe simple: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
        [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5]
    ])

    # Créer l'encodeur
    encoder = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=4
    )

    # Forward pass
    node_emb, graph_emb = encoder(x, edge_index)

    print(f"Input: x shape = {x.shape}")
    print(f"Output: node_embeddings shape = {node_emb.shape}")
    print(f"Output: graph_embedding shape = {graph_emb.shape}")

    # Vérifications
    assert node_emb.shape == (n_nodes, hidden_dim), f"Expected ({n_nodes}, {hidden_dim})"
    assert graph_emb.shape == (1, hidden_dim), f"Expected (1, {hidden_dim})"

    print(f"\nParamètres: {sum(p.numel() for p in encoder.parameters()):,}")
    print("\n✅ Test passé!")


if __name__ == "__main__":
    test_encoder()

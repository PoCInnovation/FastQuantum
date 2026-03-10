"""
Modèle Complet - Prototype v1

Pipeline:
    Graph → GNN Encoder → E_local, E_global
    problem_id → Lookup Table → E_prob
    Concat [E_global || E_local || E_prob] → Transformer → embeddings contextualisés
    Classifier binaire → probs → Symmetric BCE Loss → Backpropagation
"""

import torch
import torch.nn as nn
from encoder import GNNEncoder
from problem_embedding import ProblemEmbeddingTable
from transformer import GraphTransformer
from classifier import Classifier


class QuantumGraphModel(nn.Module):
    """
    Modèle complet pour résoudre des problèmes d'optimisation sur graphes.

    Supporte (tous binaires):
    - MaxCut (2 classes)
    - Vertex Cover (2 classes)
    - Independent Set (2 classes)

    Loss: Symmetric BCE (gère la symétrie des solutions)
    """

    def __init__(
        self,
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256,
        gnn_layers=4,
        transformer_layers=4,
        num_heads=8,
        num_problems=10,
        dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 1. GNN Encoder
        self.encoder = GNNEncoder(
            input_dim=node_input_dim,
            hidden_dim=embedding_dim,
            num_layers=gnn_layers,
            num_heads=num_heads // 2,
            dropout=dropout
        )

        # 2. Problem Embedding Table
        self.problem_embedding = ProblemEmbeddingTable(
            num_problems=num_problems,
            embedding_dim=embedding_dim
        )

        # 3. Transformer
        self.transformer = GraphTransformer(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # 4. Classifier (binaire)
        self.classifier = Classifier(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x, edge_index, problem_id, batch=None):
        """
        Args:
            x: [n_nodes, node_input_dim]
            edge_index: [2, n_edges]
            problem_id: int ou [batch_size]
            batch: [n_nodes] (optionnel)

        Returns:
            dict avec logits, probs, predictions (tous [batch, n_nodes])
        """
        # 1. GNN Encoder → E_local, E_global
        e_local, e_global = self.encoder(x, edge_index, batch)

        batch_size = e_global.shape[0]

        # Reformater e_local pour [batch_size, n_nodes, embedding_dim]
        if batch is None:
            e_local = e_local.unsqueeze(0)
        else:
            e_local = self._batch_node_embeddings(e_local, batch, batch_size)

        # 2. Problem Embedding → E_prob
        e_prob = self.problem_embedding(problem_id)
        if e_prob.shape[0] == 1 and batch_size > 1:
            e_prob = e_prob.expand(batch_size, -1)

        # 3. Transformer → embeddings contextualisés
        contextualized = self.transformer(e_local, e_global, e_prob)

        # 4. Classifier binaire → probs, predictions
        output = self.classifier(contextualized)

        return output

    def _batch_node_embeddings(self, e_local, batch, batch_size):
        """Reformate les embeddings de nœuds en [batch, max_nodes, dim]"""
        nodes_per_graph = torch.bincount(batch)
        max_nodes = nodes_per_graph.max().item()

        device = e_local.device
        out = torch.zeros(batch_size, max_nodes, self.embedding_dim, device=device)

        for b in range(batch_size):
            mask = (batch == b)
            n_nodes = mask.sum().item()
            out[b, :n_nodes] = e_local[mask]

        return out

    def compute_loss(self, logits, targets, mask=None):
        """Symmetric BCE Loss"""
        return self.classifier.compute_loss(logits, targets, mask)

    def compute_similarity(self, predictions, targets):
        """Pourcentage de ressemblance (avec symétrie)"""
        return self.classifier.compute_similarity(predictions, targets)

    def forward_with_loss(self, x, edge_index, problem_id, targets, batch=None, mask=None):
        """Forward + Loss en une seule passe"""
        output = self.forward(x, edge_index, problem_id, batch)
        loss = self.compute_loss(output['logits'], targets, mask)
        similarity = self.compute_similarity(output['predictions'], targets)
        return output, loss, similarity


if __name__ == "__main__":
    print("=== Test QuantumGraphModel ===\n")

    # Graphe simple
    n_nodes = 6
    x = torch.randn(n_nodes, 7)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
        [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5]
    ])

    # Modèle
    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256,
        gnn_layers=4,
        transformer_layers=4
    )

    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Forward (MaxCut)
    output = model(x, edge_index, problem_id=0)
    print(f"\nMaxCut:")
    print(f"  Probs: {output['probs'].shape}")
    print(f"  Predictions: {output['predictions']}")

    # Symmetric Loss
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]])
    loss = model.compute_loss(output['logits'], targets)
    print(f"  Loss: {loss.item():.4f}")

    # Test symétrie
    loss_inv = model.compute_loss(output['logits'], 1 - targets)
    print(f"  Loss inversée: {loss_inv.item():.4f}")
    print(f"  Symétrie OK ? {'✅' if abs(loss.item() - loss_inv.item()) < 1e-6 else '❌'}")

    # Similarité
    sim = model.compute_similarity(output['predictions'], targets)
    print(f"  Similarité: {sim:.0%}")

    # Backprop
    loss.backward()
    print("  Backprop OK")

    print("\n✅ Tous les tests passés!")

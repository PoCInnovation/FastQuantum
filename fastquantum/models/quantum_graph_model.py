"""
QuantumGraphModel — Full pipeline for direct solution prediction.

Pipeline:
    Graph → GNNEncoder → E_local, E_global
    problem_id → ProblemEmbeddingTable → E_prob
    Concat [E_global || E_local || E_prob] → GraphTransformer → contextualized embeddings
    Classifier → binary predictions (the solution bitstring)
"""

import torch
import torch.nn as nn

from .encoder import GNNEncoder
from .problem_embedding import ProblemEmbeddingTable
from .transformer import GraphTransformer
from .classifier import Classifier


class QuantumGraphModel(nn.Module):
    """
    Full model for direct solution prediction on combinatorial optimization problems.
    Supports MaxCut, MIS, MaxClique (all binary).
    Loss: Symmetric BCE (handles Z2 solution symmetry).
    """

    def __init__(
        self,
        node_input_dim=23,
        embedding_dim=1024,
        hidden_dim=1024,
        gnn_layers=5,
        transformer_layers=6,
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

        # 3. Graph Transformer
        self.transformer = GraphTransformer(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # 4. Binary Classifier
        self.classifier = Classifier(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x, edge_index, problem_id, batch=None):
        """
        Args:
            x: [n_nodes, node_input_dim]
            edge_index: [2, n_edges]
            problem_id: int or [batch_size]
            batch: [n_nodes] (optional for batched graphs)

        Returns:
            dict with logits, probs, predictions (all [batch, n_nodes])
        """
        # 1. GNN Encoder → E_local, E_global
        e_local, e_global = self.encoder(x, edge_index, batch)

        batch_size = e_global.shape[0]

        # Reformat e_local to [batch_size, n_nodes, embedding_dim]
        if batch is None:
            e_local = e_local.unsqueeze(0)
        else:
            e_local = self._batch_node_embeddings(e_local, batch, batch_size)

        # 2. Problem Embedding → E_prob
        e_prob = self.problem_embedding(problem_id)
        if e_prob.shape[0] == 1 and batch_size > 1:
            e_prob = e_prob.expand(batch_size, -1)

        # 3. Transformer → contextualized embeddings
        contextualized = self.transformer(e_local, e_global, e_prob)

        # 4. Binary Classifier → probs, predictions
        output = self.classifier(contextualized)

        return output

    def _batch_node_embeddings(self, e_local, batch, batch_size):
        """Reformat node embeddings to [batch, max_nodes, dim]"""
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
        """Similarity score (0-1), accounting for Z2 symmetry"""
        return self.classifier.compute_similarity(predictions, targets)

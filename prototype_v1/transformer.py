"""
Transformer pour prototype v1

Concat: [E_global || E_local || E_prob] → Transformer → embeddings contextualisés

Le Classifier est dans un fichier séparé (classifier.py)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class GraphTransformer(nn.Module):
    """
    Pour chaque nœud: input = [E_global || E_local || E_prob]
    Self-attention entre tous les nœuds.

    Output: embeddings contextualisés [batch, n_nodes, hidden_dim]
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = input_dim * 3  # E_global || E_local || E_prob

        self.input_projection = nn.Linear(concat_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, e_local, e_global, e_prob):
        """
        Args:
            e_local: [batch, n_nodes, input_dim]
            e_global: [batch, input_dim]
            e_prob: [batch, input_dim]

        Returns:
            contextualized_embeddings: [batch, n_nodes, hidden_dim]
        """
        batch_size, n_nodes, _ = e_local.shape

        # Expand pour chaque nœud
        e_global_exp = e_global.unsqueeze(1).expand(-1, n_nodes, -1)
        e_prob_exp = e_prob.unsqueeze(1).expand(-1, n_nodes, -1)

        # Concat: [E_global || E_local || E_prob]
        x = torch.cat([e_global_exp, e_local, e_prob_exp], dim=-1)

        # Projection + Positional Encoding + Transformer
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        return x  # [batch, n_nodes, hidden_dim]


if __name__ == "__main__":
    print("=== Test GraphTransformer ===")

    e_local = torch.randn(4, 6, 128)
    e_global = torch.randn(4, 128)
    e_prob = torch.randn(4, 128)

    transformer = GraphTransformer(input_dim=128, hidden_dim=256)
    output = transformer(e_local, e_global, e_prob)

    print(f"Input: e_local {e_local.shape}, e_global {e_global.shape}, e_prob {e_prob.shape}")
    print(f"Output: {output.shape}")
    print(f"Params: {sum(p.numel() for p in transformer.parameters()):,}")
    print("✅ OK")

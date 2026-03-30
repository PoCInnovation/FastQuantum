"""
Binary Classifier with Symmetric BCE Loss for FastQuantum library.

Transforms contextualized embeddings into per-node binary predictions.
The Symmetric BCE Loss handles solution symmetry (e.g. [0,1,1,0] == [1,0,0,1] for MaxCut).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Binary classifier for graph optimization problems.

    Input:  [batch, n_nodes, hidden_dim]
    Output: per-node probabilities and binary predictions [batch, n_nodes]

    Loss: Symmetric BCE
        loss = min(BCE(pred, target), BCE(pred, 1-target))
        → Handles the Z2 symmetry of binary optimization solutions
    """

    def __init__(self, hidden_dim=1024, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 1 output per node → binary
        )

    def forward(self, x):
        """
        Args:
            x: [batch, n_nodes, hidden_dim]

        Returns:
            dict with logits, probs (sigmoid), and predictions (thresholded at 0.5)
        """
        logits = self.layers(x).squeeze(-1)  # [batch, n_nodes]
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long()

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions
        }

    def compute_loss(self, logits, targets, mask=None):
        """
        Symmetric BCE Loss.
        Computes BCE in both directions and keeps the minimum.
        """
        targets = targets.float()

        if mask is not None:
            mask = mask.float()
            loss_direct = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            loss_direct = (loss_direct * mask).sum() / mask.sum().clamp(min=1)
            loss_inverse = F.binary_cross_entropy_with_logits(logits, 1.0 - targets, reduction='none')
            loss_inverse = (loss_inverse * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss_direct = F.binary_cross_entropy_with_logits(logits, targets)
            loss_inverse = F.binary_cross_entropy_with_logits(logits, 1.0 - targets)

        return torch.min(loss_direct, loss_inverse)

    def compute_similarity(self, predictions, targets):
        """
        Computes similarity score (0-1), accounting for Z2 symmetry.
        """
        predictions = predictions.float()
        targets = targets.float()
        match_direct = (predictions == targets).float().mean()
        match_inverse = (predictions == (1.0 - targets)).float().mean()
        return torch.max(match_direct, match_inverse).item()

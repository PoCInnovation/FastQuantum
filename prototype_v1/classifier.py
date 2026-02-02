"""
Classifier avec Cross-Entropy

Transforme les embeddings contextualisés en prédictions par nœud.

Input:  [batch, n_nodes, hidden_dim]
Output: [batch, n_nodes, num_classes] (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Classifier multi-classe pour les problèmes d'optimisation.

    Supporte:
    - MaxCut, Vertex Cover, Independent Set: 2 classes
    - Graph Coloring: k classes
    """

    def __init__(self, hidden_dim=256, max_classes=10, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_classes = max_classes

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_classes)
        )

    def forward(self, x, num_classes=2):
        """
        Args:
            x: [batch, n_nodes, hidden_dim] - embeddings contextualisés
            num_classes: int - nombre de classes

        Returns:
            logits: [batch, n_nodes, num_classes]
            probs: [batch, n_nodes, num_classes]
            predictions: [batch, n_nodes]
        """
        logits = self.layers(x)[:, :, :num_classes]
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions
        }

    def compute_loss(self, logits, targets, mask=None):
        """
        Cross-Entropy Loss.

        Args:
            logits: [batch, n_nodes, num_classes]
            targets: [batch, n_nodes] - classes {0, 1, ..., k-1}
            mask: [batch, n_nodes] - optionnel

        Returns:
            loss: scalar
        """
        b, n, c = logits.shape
        logits_flat = logits.reshape(-1, c)
        targets_flat = targets.reshape(-1).long()

        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

        return F.cross_entropy(logits_flat, targets_flat)


if __name__ == "__main__":
    print("=== Test Classifier ===")

    x = torch.randn(4, 6, 256)  # [batch, n_nodes, hidden_dim]
    classifier = Classifier(hidden_dim=256, max_classes=10)

    # Test 2 classes (MaxCut)
    output = classifier(x, num_classes=2)
    print(f"Logits (2 classes): {output['logits'].shape}")

    # Test 5 classes (Graph Coloring)
    output = classifier(x, num_classes=5)
    print(f"Logits (5 classes): {output['logits'].shape}")

    # Test loss
    targets = torch.randint(0, 2, (4, 6))
    output = classifier(x, num_classes=2)
    loss = classifier.compute_loss(output['logits'], targets)
    print(f"Loss: {loss.item():.4f}")

    print(f"Params: {sum(p.numel() for p in classifier.parameters()):,}")
    print("✅ OK")

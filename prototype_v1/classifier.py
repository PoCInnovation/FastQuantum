"""
Classifier Binaire avec Symmetric BCE Loss

Transforme les embeddings contextualisés en prédictions binaires par nœud.
La Symmetric BCE Loss gère automatiquement la symétrie des solutions (MaxCut).

Input:  [batch, n_nodes, hidden_dim]
Output: [batch, n_nodes] (probabilités entre 0 et 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Classifier binaire pour les problèmes d'optimisation sur graphes.

    Supporte: MaxCut, Vertex Cover, Independent Set (tous binaires).

    Loss: Symmetric BCE
        loss = min(BCE(pred, target), BCE(pred, 1-target))
        → Gère automatiquement la symétrie des solutions
    """

    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 1 seule sortie → binaire
        )

    def forward(self, x):
        """
        Args:
            x: [batch, n_nodes, hidden_dim] - embeddings contextualisés

        Returns:
            probs: [batch, n_nodes] - probabilités entre 0 et 1
            predictions: [batch, n_nodes] - 0 ou 1
        """
        logits = self.layers(x).squeeze(-1)  # [batch, n_nodes]
        probs = torch.sigmoid(logits)         # Sigmoid → [0, 1]
        predictions = (probs > 0.5).long()    # Seuil → 0 ou 1

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions
        }

    def compute_loss(self, logits, targets, mask=None):
        """
        Symmetric BCE Loss.

        Calcule la BCE dans les deux sens (target et 1-target)
        et garde le minimum → gère la symétrie.

        Args:
            logits: [batch, n_nodes] - sorties brutes (avant sigmoid)
            targets: [batch, n_nodes] - valeurs 0 ou 1
            mask: [batch, n_nodes] - optionnel (pour graphes de tailles différentes)

        Returns:
            loss: scalar
        """
        targets = targets.float()

        if mask is not None:
            mask = mask.float()

            # Loss directe : pred vs target
            loss_direct = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            loss_direct = (loss_direct * mask).sum() / mask.sum().clamp(min=1)

            # Loss inversée : pred vs (1 - target)
            loss_inverse = F.binary_cross_entropy_with_logits(
                logits, 1.0 - targets, reduction='none'
            )
            loss_inverse = (loss_inverse * mask).sum() / mask.sum().clamp(min=1)
        else:
            # Loss directe : pred vs target
            loss_direct = F.binary_cross_entropy_with_logits(logits, targets)

            # Loss inversée : pred vs (1 - target)
            loss_inverse = F.binary_cross_entropy_with_logits(logits, 1.0 - targets)

        # Symmetric : on prend le minimum des deux
        loss = torch.min(loss_direct, loss_inverse)

        return loss

    def compute_similarity(self, predictions, targets):
        """
        Calcule le pourcentage de ressemblance (en tenant compte de la symétrie).

        Args:
            predictions: [batch, n_nodes] - 0 ou 1
            targets: [batch, n_nodes] - 0 ou 1

        Returns:
            similarity: float entre 0 et 1 (1 = parfait)
        """
        predictions = predictions.float()
        targets = targets.float()

        # Ressemblance directe
        match_direct = (predictions == targets).float().mean()

        # Ressemblance inversée
        match_inverse = (predictions == (1.0 - targets)).float().mean()

        # Meilleure des deux
        similarity = torch.max(match_direct, match_inverse)

        return similarity.item()


if __name__ == "__main__":
    print("=== Test Classifier (Binaire + Symmetric BCE) ===\n")

    x = torch.randn(4, 6, 256)  # [batch, n_nodes, hidden_dim]
    classifier = Classifier(hidden_dim=256)

    # Forward
    output = classifier(x)
    print(f"Logits: {output['logits'].shape}")
    print(f"Probs: {output['probs'].shape}")
    print(f"Predictions: {output['predictions'].shape}")
    print(f"Exemple probs: {output['probs'][0].tolist()}")
    print(f"Exemple preds: {output['predictions'][0].tolist()}")

    # Test Symmetric Loss
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]] * 4).float()

    loss = classifier.compute_loss(output['logits'], targets)
    print(f"\nSymmetric BCE Loss: {loss.item():.4f}")

    # Test symétrie : target et 1-target doivent donner la même loss
    loss_normal = classifier.compute_loss(output['logits'], targets)
    loss_inverted = classifier.compute_loss(output['logits'], 1 - targets)
    print(f"Loss (target normal):  {loss_normal.item():.4f}")
    print(f"Loss (target inversé): {loss_inverted.item():.4f}")
    print(f"Égales ? {'✅ OUI' if abs(loss_normal.item() - loss_inverted.item()) < 1e-6 else '❌ NON'}")

    # Test similarité
    pred = torch.tensor([[0, 1, 0, 1, 0, 1]])
    target = torch.tensor([[1, 0, 1, 0, 1, 0]])
    sim = classifier.compute_similarity(pred, target)
    print(f"\nSimilarité [0,1,0,1,0,1] vs [1,0,1,0,1,0]: {sim:.0%}")

    print(f"\nParams: {sum(p.numel() for p in classifier.parameters()):,}")
    print("✅ OK")

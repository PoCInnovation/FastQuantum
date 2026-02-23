"""
Training Loop - Prototype v1

Pipeline:
    Graphes synthétiques → QuantumGraphModel → Symmetric BCE Loss → wandb
"""

import torch
import torch.optim as optim
import wandb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import QuantumGraphModel


# ─────────────────────────────────────────
# Hyperparamètres
# ─────────────────────────────────────────
CONFIG = {
    "epochs": 50,
    "lr": 0.001,
    "hidden_dim": 256,
    "embedding_dim": 128,
    "gnn_layers": 4,
    "transformer_layers": 4,
    "num_heads": 8,
    "n_nodes": 6,
    "problem_id": 0,   # 0=MaxCut, 1=VertexCover, 2=IndependentSet
}


def create_fake_graph(n_nodes=6, n_features=7):
    """Crée un graphe synthétique (cycle) avec un target MaxCut."""
    x = torch.randn(n_nodes, n_features)

    # Graphe en cycle : 0-1-2-3-4-5-0
    sources = list(range(n_nodes)) + list(range(1, n_nodes)) + [0]
    targets = list(range(1, n_nodes)) + [0] + list(range(n_nodes))
    edge_index = torch.tensor([sources, targets])

    # Target MaxCut : alternance 0/1 (coupe optimale pour un cycle pair)
    target = torch.tensor([[i % 2 for i in range(n_nodes)]])

    return x, edge_index, target


def train():
    # ─────────────────────────────────────
    # Init wandb
    # ─────────────────────────────────────
    wandb.init(
        project="FastQuantum",
        name="prototype-v1-maxcut",
        config=CONFIG
    )

    # ─────────────────────────────────────
    # Modèle + Optimiseur
    # ─────────────────────────────────────
    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        gnn_layers=CONFIG["gnn_layers"],
        transformer_layers=CONFIG["transformer_layers"],
        num_heads=CONFIG["num_heads"],
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params:,} paramètres")
    print(f"Epochs : {CONFIG['epochs']}")
    print(f"LR     : {CONFIG['lr']}")
    print("-" * 40)

    # Log le nombre de paramètres
    wandb.log({"n_params": n_params})

    # ─────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────
    x, edge_index, target = create_fake_graph(n_nodes=CONFIG["n_nodes"])

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # Forward + Loss + Similarité
        output, loss, similarity = model.forward_with_loss(
            x=x,
            edge_index=edge_index,
            problem_id=CONFIG["problem_id"],
            targets=target
        )

        # Backward
        loss.backward()
        optimizer.step()

        # ─────────────────────────────────
        # Log wandb
        # ─────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "similarity": similarity,
        })

        # Affichage terminal
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{CONFIG['epochs']} │ "
                  f"Loss: {loss.item():.4f} │ "
                  f"Similarité: {similarity:.0%}")

    print("-" * 40)
    print("Entraînement terminé !")
    print(f"Loss finale    : {loss.item():.4f}")
    print(f"Similarité finale : {similarity:.0%}")
    print(f"Prédictions : {output['predictions'].tolist()}")
    print(f"Target      : {target.tolist()}")

    wandb.finish()


if __name__ == "__main__":
    train()

"""
Training Loop - Prototype v1

Pipeline:
    Dataset réel → QuantumGraphModel → Symmetric BCE Loss → wandb

Utilisation :
    python train.py                          # config par défaut
    python train.py --lr 0.0001             # learning rate différent
    python train.py --lr 0.0001 --hidden_dim 128 --epochs 100
"""

import torch
import torch.optim as optim
import wandb
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import QuantumGraphModel
from dataset import load_dataloaders

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset_v1.json", "dataset_v1.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train QuantumGraphModel")

    parser.add_argument("--epochs",             type=int,   default=10)
    parser.add_argument("--lr",                 type=float, default=0.001)
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--embedding_dim",      type=int,   default=128)
    parser.add_argument("--gnn_layers",         type=int,   default=4)
    parser.add_argument("--transformer_layers", type=int,   default=4)
    parser.add_argument("--num_heads",          type=int,   default=8)

    return parser.parse_args()


def train(config):
    # ─────────────────────────────────────
    # Init wandb
    # ─────────────────────────────────────
    wandb.init(project="FastQuantum", config=config)

    # ─────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────
    train_loader, val_loader, _ = load_dataloaders(DATASET_PATH)

    # ─────────────────────────────────────
    # Device (GPU si disponible, sinon CPU)
    # ─────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ─────────────────────────────────────
    # Modèle + Optimiseur
    # ─────────────────────────────────────
    model = QuantumGraphModel(
        node_input_dim=23,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        gnn_layers=config["gnn_layers"],
        transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params:,} paramètres")
    print(f"Epochs : {config['epochs']}  |  LR : {config['lr']}  |  hidden_dim : {config['hidden_dim']}")
    print("-" * 40)
    wandb.log({"n_params": n_params})

    # ─────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        model.train()

        epoch_loss = 0
        epoch_similarity = 0

        for batch in train_loader:
            optimizer.zero_grad()

            output, loss, similarity = model.forward_with_loss(
                x=batch.x.to(device),
                edge_index=batch.edge_index.to(device),
                problem_id=batch.problem_id.item(),
                targets=batch.y.unsqueeze(0).to(device)
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_similarity += similarity

        # Moyennes sur l'epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_sim  = epoch_similarity / len(train_loader)

        # ─────────────────────────────────
        # Validation
        # ─────────────────────────────────
        model.eval()
        val_loss = 0
        val_similarity = 0

        with torch.no_grad():
            for batch in val_loader:
                output, loss, similarity = model.forward_with_loss(
                    x=batch.x.to(device),
                    edge_index=batch.edge_index.to(device),
                    problem_id=batch.problem_id.item(),
                    targets=batch.y.unsqueeze(0).to(device)
                )
                val_loss += loss.item()
                val_similarity += similarity

        avg_val_loss = val_loss / len(val_loader)
        avg_val_sim  = val_similarity / len(val_loader)

        # Log wandb
        wandb.log({
            "epoch":          epoch,
            "train_loss":     avg_loss,
            "train_similarity": avg_sim,
            "val_loss":       avg_val_loss,
            "val_similarity": avg_val_sim,
        })

        print(f"Epoch {epoch:2d}/{config['epochs']} │ "
              f"Train Loss: {avg_loss:.4f} Sim: {avg_sim:.0%} │ "
              f"Val Loss: {avg_val_loss:.4f} Sim: {avg_val_sim:.0%}")

    print("-" * 40)
    print("Entraînement terminé !")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    train(config)

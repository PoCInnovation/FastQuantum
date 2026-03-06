"""
Wandb Sweep - Prototype v1

Lance automatiquement plusieurs runs avec différents hyperparamètres
et compare les résultats sur le dashboard wandb.

Utilisation :
    python sweep.py
"""

import wandb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import train


# ─────────────────────────────────────────
# Configuration du sweep
# ─────────────────────────────────────────
SWEEP_CONFIG = {
    "name": "FastQuantum-sweep",
    "method": "bayes",   # bayes > grid (648 combinaisons possibles, on laisse wandb choisir les meilleurs)
    "metric": {
        "name": "val_similarity",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {
            "values": [0.01, 0.001, 0.0001]
        },
        "hidden_dim": {
            "values": [128, 256]
        },
        "gnn_layers": {
            "values": [2, 4]
        },
        "transformer_layers": {
            "values": [2, 4, 6]
        },
        "embedding_dim": {
            "values": [64, 128, 256]
        },
        "num_heads": {
            "values": [4, 8]
        },
        "weight_decay": {
            "values": [0, 1e-4, 1e-5]
        }
    }
}


def train_sweep():
    """Un run du sweep : wandb injecte les hyperparamètres puis appelle train()"""
    wandb.init()
    config = dict(wandb.config)   # hyperparamètres injectés par wandb

    # Valeurs fixes non testées dans le sweep
    config.setdefault("epochs", 50)

    print(f"\nRun : lr={config['lr']}, hidden_dim={config['hidden_dim']}, gnn_layers={config['gnn_layers']}, "
          f"transformer_layers={config['transformer_layers']}, embedding_dim={config['embedding_dim']}, "
          f"num_heads={config['num_heads']}, weight_decay={config['weight_decay']}")

    train(config)


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="FastQuantum")

    print(f"\nSweep ID : {sweep_id}")
    print("Méthode : bayes (648 combinaisons possibles, wandb choisit intelligemment)")
    print("Lancement des runs...\n")

    wandb.agent(sweep_id, function=train_sweep)

"""
Dataset et DataLoader - Prototype v1

Charge le dataset JSON et le convertit en tenseurs PyTorch Geometric.

Structure d'un sample JSON :
    - problem       : "MIS", "MAXCUT", "MAX_CLIQUE"
    - n_nodes       : nombre de nœuds
    - adj           : matrice d'adjacence [n_nodes x n_nodes]
    - x             : features des nœuds [n_nodes x 23]
    - exact_solution: target bitstring [n_nodes] → {0, 1}

Utilisation :
    from dataset import load_dataloaders
    train_loader, val_loader, test_loader = load_dataloaders("dataset_v1.json")
"""

import json
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


# Mapping problème → ID
PROBLEM_TO_ID = {
    "MAXCUT":    0,
    "MIS":       1,
    "MAX_CLIQUE": 2,
}


def adj_to_edge_index(adj):
    """
    Convertit une matrice d'adjacence en edge_index (format PyTorch Geometric).

    Matrice adj [n x n] → edge_index [2, n_edges]

    Exemple :
        adj = [[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]]

        edge_index = [[0, 1, 1, 2],
                      [1, 0, 2, 1]]
    """
    adj_tensor = torch.tensor(adj, dtype=torch.float)
    edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()
    return edge_index


class GraphDataset(Dataset):
    """
    Dataset PyTorch pour les graphes d'optimisation.

    Chaque sample contient :
        - data.x          : features des nœuds [n_nodes, 23]
        - data.edge_index : connexions [2, n_edges]
        - data.y          : target bitstring [n_nodes]
        - data.problem_id : int (0=MAXCUT, 1=MIS, 2=MAX_CLIQUE)
        - data.n_nodes    : int
    """

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.samples = []

        for item in raw_data:
            # Features des nœuds [n_nodes, 23]
            x = torch.tensor(item["x"], dtype=torch.float)

            # Connexions : adj → edge_index
            edge_index = adj_to_edge_index(item["adj"])

            # Target : exact_solution [n_nodes]
            y = torch.tensor(item["exact_solution"], dtype=torch.long)

            # ID du problème
            problem_id = PROBLEM_TO_ID[item["problem"]]

            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                problem_id=problem_id,
                n_nodes=item["n_nodes"]
            )
            self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_dataloaders(json_path, batch_size=1, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Charge le dataset et retourne 3 DataLoaders (train / val / test).

    Split :
        80% → train
        10% → val
        10% → test

    Args:
        json_path  : chemin vers le fichier JSON
        batch_size : taille des batchs (défaut=1 car graphes de tailles différentes)
        train_ratio: proportion du train
        val_ratio  : proportion de la validation
        seed       : graine aléatoire pour reproductibilité

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = GraphDataset(json_path)
    total = len(dataset)

    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    print(f"Dataset chargé : {total} samples")
    print(f"  Train : {n_train} ({train_ratio:.0%})")
    print(f"  Val   : {n_val}   ({val_ratio:.0%})")
    print(f"  Test  : {n_test}  ({1 - train_ratio - val_ratio:.0%})")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import os

    json_path = os.path.join(
        os.path.dirname(__file__),
        "dataset_v1.json", "dataset_v1.json"
    )

    print("=== Test DataLoader ===\n")

    train_loader, val_loader, test_loader = load_dataloaders(json_path)

    print("\nPremier sample du train :")
    sample = next(iter(train_loader))
    print(f"  x          : {sample.x.shape}")
    print(f"  edge_index : {sample.edge_index.shape}")
    print(f"  y (target) : {sample.y}")
    print(f"  problem_id : {sample.problem_id}")
    print(f"  n_nodes    : {sample.n_nodes}")
    print(f"  Nœuds      : {sample.x.shape[0]}")

    print("\nDistribution des problèmes (train) :")
    from collections import Counter
    problems = Counter(
        batch.problem_id.item()
        for batch in train_loader
    )
    id_to_problem = {v: k for k, v in PROBLEM_TO_ID.items()}
    for pid, count in sorted(problems.items()):
        print(f"  {id_to_problem[pid]} : {count} samples")

    print("\nOK DataLoader")

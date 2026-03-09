# QuantumGraphModel

Modèle GNN-Transformer hybride pour la résolution de problèmes d'optimisation combinatoire sur graphes (NP-difficiles).

## Problèmes supportés

| Problème | ID | Description |
|----------|----|-------------|
| MaxCut | 0 | Partitionner les nœuds en 2 groupes pour maximiser les arêtes coupées |
| Vertex Cover | 1 | Trouver le minimum de nœuds couvrant toutes les arêtes |
| Independent Set | 2 | Trouver le maximum de nœuds sans arêtes communes |

## Architecture

```
Graphe (x, edge_index)
        │
        ▼
┌───────────────┐
│  GNN Encoder  │  GAT multi-head → E_local [n_nodes, 128]
│  (encoder.py) │                   E_global [batch, 128]
└───────┬───────┘
        │
problem_id ──→ ┌──────────────────────┐
               │ Problem Embedding    │ → E_prob [batch, 128]
               │ (problem_embedding.py│
               └──────────┬───────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  Concat [E_global ║ E_local ║ E_prob]  │  → [batch, n_nodes, 384]
        │  Graph Transformer (transformer.py) │  → [batch, n_nodes, 256]
        └─────────────────┬──────────────────┘
                          │
               ┌──────────▼──────────┐
               │  Classifier Binaire │ Sigmoid → probs [batch, n_nodes]
               │  (classifier.py)    │ Seuil   → predictions {0, 1}
               └─────────────────────┘
```

## Loss : Symmetric BCE

Gère automatiquement la symétrie des solutions (ex: MaxCut).

```
loss = min(BCE(pred, target), BCE(pred, 1-target))
```

`[0,1,0,1]` et `[1,0,1,0]` donnent la **même loss** car ce sont deux représentations équivalentes de la même coupe.

## Structure des fichiers

### `encoder.py` — GNN Encoder
Encode la structure du graphe via des couches **GAT** (Graph Attention Network).
- Prend en entrée les features des nœuds `x` et les connexions `edge_index`
- Chaque nœud agrège l'information de ses voisins avec des **scores d'attention** (qui écouter et combien)
- Produit deux sorties :
  - `E_local [n_nodes, 128]` — embedding par nœud (vision locale)
  - `E_global [batch, 128]` — embedding du graphe entier via mean pooling (vision globale)

### `problem_embedding.py` — Problem Embedding Table
Transforme un `problem_id` (entier) en vecteur appris.
- Fonctionne comme un dictionnaire : `0 → vecteur MaxCut`, `1 → vecteur VertexCover`, etc.
- Le vecteur est **appris** pendant l'entraînement (backpropagation)
- Produit `E_prob [batch, 128]` — contexte du problème à résoudre

### `transformer.py` — Graph Transformer
Combine les trois embeddings et applique une **self-attention globale** entre tous les nœuds.
- Concat : `[E_global || E_local || E_prob]` → `[batch, n_nodes, 384]`
- Projection linéaire → `[batch, n_nodes, 256]`
- Positional Encoding + Transformer Encoder (chaque nœud peut voir tous les autres)
- Produit des embeddings **contextualisés** `[batch, n_nodes, 256]`

### `classifier.py` — Classifier Binaire + Symmetric BCE Loss
Prédit un label `{0, 1}` pour chaque nœud.
- Linear → GELU → Linear → Sigmoid → probabilité entre 0 et 1
- **Symmetric BCE Loss** : `min(BCE(pred, target), BCE(pred, 1-target))`
  - Gère la symétrie des solutions (MaxCut : `[0,1,0,1]` = `[1,0,1,0]`)
- **compute_similarity()** : pourcentage de ressemblance en tenant compte de la symétrie

### `model.py` — QuantumGraphModel (assemblage complet)
Assemble les 4 composants dans un seul `nn.Module`.
- `forward()` : pipeline complet graphe → prédictions
- `forward_with_loss()` : forward + loss + similarité en une passe
- `compute_loss()` : Symmetric BCE Loss
- `compute_similarity()` : métrique de ressemblance

### `train.py` — Training Loop
Boucle d'entraînement avec intégration **wandb**.
- Hyperparamètres centralisés dans `CONFIG`
- Log à chaque epoch : `loss`, `similarity`
- Dashboard disponible sur [wandb.ai](https://wandb.ai) (projet: FastQuantum)

### `test_model.py` — Tests Unitaires
10 tests couvrant chaque composant et le modèle complet.
- Test de chaque composant individuellement (encoder, transformer, classifier...)
- Test de la **Symmetric BCE Loss** (vérifie que target et 1-target donnent la même loss)
- Test de la **métrique de similarité**
- Test de la **backpropagation** (vérifie que les gradients existent partout)
- Test avec différentes tailles de graphes (4, 8, 16, 32 nœuds)

## Format des données

### Entrée

```python
x          = torch.Tensor  # [n_nodes, 7]  - features des nœuds
edge_index = torch.Tensor  # [2, n_edges]  - connexions (format PyG)
problem_id = int           # 0, 1 ou 2
```

### Sortie

```python
{
    'logits':      # [batch, n_nodes]    - sorties brutes
    'probs':       # [batch, n_nodes]    - probabilités [0, 1]
    'predictions': # [batch, n_nodes]    - 0 ou 1
}
```

### Target (pour l'entraînement)

```python
target = torch.Tensor  # [batch, n_nodes] - bitstring {0, 1} généré par QAOA
```

## Utilisation

### Inférence

```python
from model import QuantumGraphModel

model = QuantumGraphModel()

output = model(x, edge_index, problem_id=0)
print(output['predictions'])  # [1, 0, 1, 0, 1, 0]
```

### Entraînement

```python
output, loss, similarity = model.forward_with_loss(
    x=x,
    edge_index=edge_index,
    problem_id=0,
    targets=target
)
loss.backward()
optimizer.step()
```

### Lancer les tests

```bash
python test_model.py
# → 10/10 tests passés
```

### Lancer l'entraînement

```bash
python train.py
# → Logs sur wandb (projet: FastQuantum)
```

## Paramètres du modèle

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `node_input_dim` | 7 | Nombre de features par nœud |
| `embedding_dim` | 128 | Dimension des embeddings (E_local, E_global, E_prob) |
| `hidden_dim` | 256 | Dimension interne du Transformer et Classifier |
| `gnn_layers` | 4 | Nombre de couches GAT |
| `transformer_layers` | 4 | Nombre de couches Transformer |
| `num_heads` | 8 | Têtes d'attention |

**Total : ~3.3M paramètres**

## Dépendances

```bash
pip install torch torch-geometric wandb
```

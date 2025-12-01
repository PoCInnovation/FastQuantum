# GNN Fast Quantum - QAOA Parameter Prediction

Prédiction rapide des paramètres QAOA optimaux pour des problèmes d'optimisation combinatoire quantique en utilisant des Graph Neural Networks (GNN).

## 📁 Structure du projet

### `test_gnn_with_qiskit.py` - Validation du modèle
Script de test pour valider les prédictions du GNN contre les optimisations QAOA réelles de Qiskit.

**Usage:**
```bash
python test_gnn_with_qiskit.py
```
**prérequis:**
- si mac os alors->   --no-gpu \
- si on veut utilisé les GPU->  linux nécéssaire

**Ce qu'il fait:**
- Charge le modèle GNN entraîné (`FastQuantum/best_qaoa_gat_model.pt`)
- Génère des graphes de test
- Prédit gamma/beta avec le GNN (instantané)
- Compare avec l'optimal trouvé par Qiskit QAOA (lent)
- Affiche les gaps de performance

**Résultats attendus:**
- Gap moyen: ~5% (excellent)
- Speedup: 100-1000x plus rapide que QAOA

---

### `Generator(forGPU)/` - Génération de dataset quantique

Générateur de dataset multi-problèmes avec support GPU et parallélisation CPU.

**Fichiers principaux:**
- `gen_quantum_mp.py`: Générateur multi-problèmes optimisé
- `check_gpu_availability.py`: Diagnostic GPU/CUDA

**Problèmes supportés:**
- MaxCut (partition de graphe)
- Vertex Cover (couverture minimale)
- Max Independent Set (sélection de nœuds)
- Graph Coloring (3 couleurs avec 2 qubits par nœud) 

warning :
Graph Coloring utilise 2× plus de qubits (2 par nœud au lieu de 1):
Graphe 10 nœuds = 20 qubits pour coloring vs 10 pour MaxCut
-> donc Simulation BEAUCOUP plus lente (~5-10x plus lent)
Recommandation: Limitez graph_coloring à 10% du dataset et utilisez des petits graphes (5-8 nœuds) pour ce problème.

warning2 :
Warn start utilise des données heuristique mais c'ets juste une optimisation de temps gratuites -> pas de baisse de qulité du dataset ici

**Usage:**
```bash
# Test GPU disponible
python Generator(forGPU)/check_gpu_availability.py

# Générer dataset (CPU)
python Generator(forGPU)/gen_quantum_mp.py \
  --samples 200 \
  --min_nodes 6 \
  --max_nodes 12 \
  --maxcut-ratio 0.70 \
  --workers 12 \
  --no-gpu \
  --output Dataset/phase1_hybrid70.json

# Générer dataset (GPU - Linux)
python Generator(forGPU)/gen_quantum_mp.py \
  --samples 200 \
  --min_nodes 6 \
  --max_nodes 12 \
  --maxcut-ratio 0.70 \
  --workers 8 \
  --output Dataset/phase1_hybrid70.json
```

**Arguments:**
- `--samples`: Nombre d'échantillons à générer
- `--min_nodes`, `--max_nodes`: Taille des graphes
- `--maxcut-ratio`: Ratio de MaxCut (ex: 0.70 = 70% MaxCut, 30% autres)
- `--workers`: Nombre de workers parallèles
- `--no-gpu`: Forcer CPU (Windows)
- `--checkpoint`: Sauvegarder tous les N samples

**Performance:**
- **GPU (Linux)**: ~5-15s par sample (recommandé)
- **CPU (16 cores)**: ~20-40s par sample
- **CPU (4 cores)**: ~60-120s par sample

---

## 🚀 Quick Start

### 1. Tester le modèle existant
```bash
python test_gnn_with_qiskit.py
```

### 2. Générer un nouveau dataset
```bash
# Windows (CPU seulement)
python Generator(forGPU)/gen_quantum_mp.py --samples 50 --workers 4 --no-gpu

# Linux avec GPU
python Generator(forGPU)/gen_quantum_mp.py --samples 200 --workers 8
```

### 3. Entraîner le modèle (dans FastQuantum/)
```bash
python FastQuantum/GnnmodelGat.py
```
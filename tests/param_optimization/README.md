# Param Optimization - QAOA Dataset Generation & Evaluation

This directory contains a complete pipeline for generating and evaluating QAOA parameter optimization datasets **with real Qiskit execution**.

## 🎯 Objective

Create a high-quality dataset linking:
- **Graph characteristics** (size, density, clustering, etc.)
- **QAOA parameters** (depth p, initial angles β/γ, optimizer)
- **Measured performance** (final energy, approximation ratio, convergence)

This dataset is then used to train ML models capable of **predicting the best QAOA parameters** for a new graph.

## 📁 Files

- `generate_dataset.py`: Dataset generation with Qiskit QAOA (338 lines)
- `evaluate_dataset.py`: Statistical analysis and dataset validation
- `train_baseline.py`: Baseline ML model training (RandomForest, XGBoost)
- `requirements.txt`: Python dependencies
- `setup.sh`: Automatic installation script
- `SIMPLIFICATION.md`: Code improvement documentation

## 🚀 Installation

```bash
# Option 1: Automatic installation
cd tests/param_optimization
bash setup.sh

# Option 2: Manual installation
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 📊 Usage

### 1. Generate a dataset with Qiskit QAOA

```bash
# Quick test dataset (2 minutes)
python generate_dataset.py \
    --n_graphs 10 \
    --samples_per_graph 5 \
    --out quick_test.parquet

# Research dataset (recommended: 500-1000 graphs)
python generate_dataset.py \
    --n_graphs 500 \
    --samples_per_graph 10 \
    --n_min 6 --n_max 12 \
    --density_min 0.3 --density_max 0.7 \
    --p_choices 1,2,3 \
    --graph_types erdos_renyi,barabasi_albert \
    --out research_dataset.parquet \
    --seed 42
```

**Note**: The script **always** uses real QAOA with Qiskit. Estimated time: ~3-5s per sample.

**Key parameters:**
- `--n_graphs`: Number of random graphs (recommended: 100-1000)
- `--samples_per_graph`: QAOA configurations per graph (recommended: 5-20)
- `--n_min/n_max`: Graph size range (6-12 recommended)
- `--density_min/max`: Density variability (0.3-0.7 recommended)
- `--p_choices`: QAOA depths to test (e.g., 1,2,3)
- `--graph_types`: Graph types (erdos_renyi, barabasi_albert, watts_strogatz, regular)
- `--seed`: Seed for reproducibility

### 2. Evaluate dataset quality

```bash
python evaluate_dataset.py research_dataset.parquet
```

Displays:
- Descriptive statistics (sizes, distributions)
- Metric variability (crucial for ML!)
- Correlations
- Improvement recommendations

### 3. Test with ML baseline

```bash
python train_baseline.py dataset.parquet \
    --target approximation_ratio \
    --test_size 0.2
```

Trains RandomForest and XGBoost to predict QAOA performance.
Metrics: RMSE, R², Spearman rank correlation.

**Interpretation:**
- R² > 0.5 → **Excellent signal**, very useful dataset
- R² 0.2-0.5 → Moderate signal, improve variability
- R² < 0.2 → Poor dataset, revise protocol

## 📈 Improving the Dataset

### For a better dataset:

1. **Increase size**: Minimum 1000-5000+ experiments
2. **Diversify graphs**:
   - Multiple sizes (n=4 to 16+)
   - Multiple densities (0.2 to 0.8)
   - Multiple types (ER, BA, WS, regular, planted)
3. **Use real QAOA**: Qiskit for authentic measurements
4. **Vary parameters**: Multiple p, optimizers, initial angles
5. **Repeat with seeds**: Multiple runs per graph for robustness

### Example production command:

```bash
python generate_dataset.py \
    --n_graphs 500 \
    --samples_per_graph 10 \
    --n_min 4 --n_max 14 \
    --density_min 0.2 --density_max 0.8 \
    --p_choices 1,2,3,4 \
    --graph_types erdos_renyi,barabasi_albert,watts_strogatz \
    --out large_dataset.parquet \
    --seed 42
```

(Estimated time: several hours with real QAOA)

## 🔬 Experimental Protocol

### Dataset structure (columns):

**Identifiers:**
- `graph_id`, `seed`

**Graph features:**
- `num_nodes`, `num_edges`, `density`
- `degree_mean`, `degree_std`
- `clustering_coeff`, `assortativity`
- `graph_type`

**QAOA parameters:**
- `p` (depth)
- `init_beta`, `init_gamma` (initial angles, JSON)
- `optimizer`

**Performance metrics:**
- `final_energy` (Hamiltonian energy)
- `cut_value` (cut size found)
- `approximation_ratio` (quality vs optimal)
- `success_prob` (quality proxy)
- `iterations` (optimizer iteration count)
- `converged` (binary convergence)
- `runtime` (execution time)
- `optimal_cut` (exact solution if computable)

## 🧪 Next Steps

1. **Integrate Vertex Cover** and other combinatorial problems
2. **Warm-start angles** based on heuristics
3. **Meta-learning**: Transfer parameters between similar graphs
4. **Bandits**: Adaptive parameter sampling
5. **Neural architecture**: GNN for graph encoding, MLP for parameter prediction

## 📚 References

- QAOA: Farhi & Goldstone (2014)
- Dataset best practices: ML for Quantum Computing
- Graph features: NetworkX documentation



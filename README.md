# FastQuantum

FastQuantum is a research project exploring the intersection of **machine learning** and **quantum computing**.
Its main objective is to develop an AI system capable of **predicting the optimal parameters** for efficiently running quantum algorithms.
In the long term, the ambition is to go even further by creating a model able to **predict quantum algorithm results themselves**—a challenging goal that remains out of reach for now but guides the project’s future direction.

## How does it work?

FastQuantum currently focuses on using **Graph Neural Networks (GNNs)** and **Quantum Neural Networks (QNNs)** to learn how to predict optimal parameters for quantum algorithms.

Many quantum algorithms—such as **MaxCut** or **Vertex Cover**—can be represented as graphs. This makes GNNs a natural fit: they can capture the structure of the problem instance and learn meaningful patterns directly from the graph topology. In parallel, QNNs allow the model to integrate quantum-inspired representations that may generalize better to circuits with quantum-specific behavior.
## Key Features

### Graph Neural Network (GNN)
We utilize a **Graph Attention Network (GAT)** to process the graph-structured data.
- **Edge-Conditioned Attention**: The model explicitly uses edge weights ($J_{ij}$ from the Hamiltonian) to determine the importance of connections, mimicking the physical interactions of the MaxCut problem.
- **Laplacian Positional Encodings (LPE)**: We inject the "spectral coordinates" of nodes (eigenvectors of the Laplacian) as features. This gives the GNN a sense of "GPS" to understand the graph's geometry and symmetry, significantly improving its ability to distinguish non-isomorphic graphs.
- **Input Normalization**: A Batch Normalization layer ensures that these rich features are properly scaled for efficient learning.

## Getting Started

### Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/PoCInnovation/FastQuantum.git
cd FastQuantum
pip install -r requirements.txt
```

### Usage

1. **Generate Dataset**
   Create a training and validation dataset with enriched features (heuristics + LPE).
   ```bash
   python generate_dataset.py
   ```
   *This will create `qaoa_train_dataset.json` and `qaoa_val_dataset.json` in the `Dataset/` folder.*

2. **Train Model**
   Train the GAT model on the generated data.
   ```bash
   python GnnmodelGat.py
   ```
   *The script will automatically detect the GPU, train the model, and save the best checkpoint as `best_qaoa_gat_model.pt`.*

## Dataset Details

### Generation Script (`generate_v1_dataset.py`)

The dataset generation script is designed to create diverse graph instances and find optimal QAOA parameters for them. It supports **MaxCut**, **Maximum Independent Set (MIS)**, and **Maximum Clique** problems.

**Key Steps:**
1.  **Graph Generation**: Creates graphs using various topologies (Erdős-Rényi, Barabási-Albert, Regular, Watts-Strogatz, Lollipop) to ensure diversity.
2.  **Feature Extraction**: Computes node features including degree, centrality measures (betweenness, closeness, eigenvector, PageRank), clustering coefficients, and Random Walk Positional Encodings (RWPE).
3.  **Ground Truth Calculation**: Solves the problem exactly using brute force to establish a baseline.
4.  **QAOA Optimization**: Uses Qiskit's `StatevectorEstimator` and COBYLA optimizer to find optimal `gamma` and `beta` parameters that maximize the expected energy.
5.  **Filtering**: Only saves instances where the QAOA solution achieves a high approximation ratio (e.g., > 0.85) relative to the optimal solution.

**Usage Arguments:**
- `--problem`: The optimization problem to solve (`MAXCUT`, `MIS`, `MAX_CLIQUE`). Default: `MAXCUT`.
- `--samples`: Target number of samples to generate. Default: `100`.
- `--nodes`: Range of nodes (format "min-max", e.g. `8-16`). Default: `8-16`.
- `--workers`: Number of parallel processes to use. Default: `os.cpu_count()-1`.
- `--output`: Path to save the generated JSON file. Default: `Dataset/qaoa_dataset.json`.

**Example Command:**
```bash
python generate_v1_dataset.py --problem MIS --samples 500 --nodes 10-14 --output Dataset/train_mis.json
```

### Output Data Structure

The output is a JSON file containing a list of graph instances. Each instance is a dictionary with the following fields:

*   **`id`** *(int)*: Unique identifier for the sample within the dataset.
*   **`problem`** *(str)*: The type of combinatorial problem (e.g., "MIS").
*   **`n_nodes`** *(int)*: Number of nodes in the graph.
*   **`topo`** *(str)*: The generator used for the graph topology (e.g., "BA" for Barabási-Albert).
*   **`adj`** *(List[List[float]])*: The adjacency matrix of the graph. $A_{ij} = 1.0$ if an edge exists between node $i$ and $j$, else $0.0$.
*   **`x`** *(List[List[float]])*: Matrix of node features. Each row corresponds to a node and contains a concatenated vector of:
    *   Node heuristics (Degree, Degree Centrality, Clustering Coeff, Betweenness, Closeness, PageRank, Eigenvector Centrality).
    *   RWPE (Random Walk Positional Encodings) features (default depth of 16).
*   **`gamma`** *(List[float])*: Optimized QAOA rotation angles for the phase separator unitary ($U_C(\gamma)$).
*   **`beta`** *(List[float])*: Optimized QAOA rotation angles for the mixing unitary ($U_B(\beta)$).
*   **`ratio`** *(float)*: The approximation ratio achieved by the QAOA parameters compared to the exact brute-force solution.
*   **`exact_value`** *(float)*: The optimal solution value found by brute force (e.g., max independent set size).
*   **`exact_solution`** *(List[int])*: The bitstring configuration achieving the optimal value (e.g., `[0, 1, 0, 1]` where 1 means node selected).

#### Example JSON Object

```json
{
  "id": 0,
  "problem": "MIS",
  "n_nodes": 14,
  "topo": "BA",
  "adj": [
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, ...], 
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
    ...
  ],
  "x": [
    [6.0, 0.46, 0.20, 0.27, 0.65, 0.12, 0.39, ...], // Features for Node 0
    [1.0, 0.07, 0.00, 0.00, 0.40, 0.02, 0.09, ...]  // Features for Node 1
  ],
  "gamma": [3.380012773071252],
  "beta": [0.5386966770357803],
  "ratio": 0.973516949739922,
  "exact_value": 6.0,
  "exact_solution": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
}
```

## Get involved

You're invited to join this project ! Check out the [contributing guide](./CONTRIBUTING.md).

If you're interested in how the project is organized at a higher level, please contact the current project manager.

## Our PoC team ❤️

Developers
| [<img src="https://avatars.githubusercontent.com/u/146193362?v=4?size=85" width=85><br><sub>Elie Stroun</sub>](https://github.com/eliestroun14) | [<img src="https://avatars.githubusercontent.com/u/141178010?v=4?size=85" width=85><br><sub>Gregroire Caseaux</sub>](https://github.com/Nezketsu) | [<img src="https://avatars.githubusercontent.com/u/94183376?v=4?size=85" width=85><br><sub>Noa Smoter</sub>](https://github.com/Nerzouille) | [<img src="https://avatars.githubusercontent.com/u/101893788?v=4?size=85" width=85><br><sub>Pierre Beaud</sub>](https://github.com/divisio74)
| :---: | :---: | :---: | :---: |

Manager
| [<img src="https://avatars.githubusercontent.com/u/52128884?v=4?size=85" width=85><br><sub>Sacha Henneveux</sub>](https://github.com/SachaHenneveux)
| :---: |

<h2 align=center>
Organization
</h2>

<p align='center'>
    <a href="https://www.linkedin.com/company/pocinnovation/mycompany/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn logo">
    </a>
    <a href="https://www.instagram.com/pocinnovation/">
        <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram logo"
>
    </a>
    <a href="https://twitter.com/PoCInnovation">
        <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter logo">
    </a>
    <a href="https://discord.com/invite/Yqq2ADGDS7">
        <img src="https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord logo">
    </a>
</p>
<p align=center>
    <a href="https://www.poc-innovation.fr/">
        <img src="https://img.shields.io/badge/WebSite-1a2b6d?style=for-the-badge&logo=GitHub Sponsors&logoColor=white" alt="Website logo">
    </a>
</p>

> 🚀 Don't hesitate to follow us on our different networks, and put a star 🌟 on `PoC's` repositories

> Made with ❤️ by PoC

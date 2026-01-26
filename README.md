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
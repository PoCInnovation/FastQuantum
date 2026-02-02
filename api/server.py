#!/usr/bin/env python3
"""
FastAPI server for QAOA GNN model inference.
Provides endpoints for graph generation and parameter prediction.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GnnmodelGat import QAOAPredictorGAT

app = FastAPI(title="FastQuantum API", version="1.0.0")

# CORS for frontend - allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global model instance
model = None
device = None
p_layers = 1
input_dim = 7


def load_model():
    """Load the trained model."""
    global model, device, p_layers, input_dim

    model_path = Path(__file__).parent.parent / "best_qaoa_gat_model.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    p_layers = checkpoint['p_layers']

    model = QAOAPredictorGAT(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        p_layers=p_layers,
        attention_heads=8,
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded on {device}: {input_dim} features, p={p_layers}")


def compute_node_features(G: nx.Graph) -> np.ndarray:
    """Compute 7 node features for the graph."""
    n_nodes = G.number_of_nodes()
    features = np.zeros((n_nodes, 7))

    degrees = dict(G.degree())
    features[:, 0] = [degrees[i] for i in range(n_nodes)]

    degree_centrality = nx.degree_centrality(G)
    features[:, 1] = [degree_centrality[i] for i in range(n_nodes)]

    clustering = nx.clustering(G)
    features[:, 2] = [clustering[i] for i in range(n_nodes)]

    betweenness = nx.betweenness_centrality(G)
    features[:, 3] = [betweenness[i] for i in range(n_nodes)]

    closeness = nx.closeness_centrality(G)
    features[:, 4] = [closeness[i] for i in range(n_nodes)]

    pagerank = nx.pagerank(G, max_iter=1000)
    features[:, 5] = [pagerank[i] for i in range(n_nodes)]

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        features[:, 6] = [eigenvector[i] for i in range(n_nodes)]
    except:
        features[:, 6] = 0.0

    return features


def graph_to_pyg_data(G: nx.Graph) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data."""
    node_features = compute_node_features(G)
    x = torch.tensor(node_features, dtype=torch.float)

    adj_matrix = nx.to_numpy_array(G)
    edge_index = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long)

    y = torch.zeros(p_layers * 2)

    return Data(x=x, edge_index=edge_index, y=y)


def get_node_importance(G: nx.Graph, data: Data) -> List[float]:
    """Calculate node importance using GAT attention weights."""
    n_nodes = G.number_of_nodes()

    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch_data = next(iter(loader)).to(device)

    attention_weights = model.get_attention_weights(batch_data)

    node_importance = np.zeros(n_nodes)

    for edge_idx, alpha in attention_weights:
        alpha_var = alpha.var(dim=1).numpy()
        alpha_mean = alpha.mean(dim=1).numpy()
        alpha_score = alpha_mean * (1 + alpha_var * 10)

        edge_idx = edge_idx.numpy()

        for i in range(edge_idx.shape[1]):
            src = edge_idx[0, i]
            dst = edge_idx[1, i]
            node_importance[src] += alpha_score[i]
            node_importance[dst] += alpha_score[i]

    if node_importance.max() > node_importance.min():
        node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
    else:
        node_importance = np.ones(n_nodes)

    return node_importance.tolist()


# Request/Response models
class GenerateRequest(BaseModel):
    n_nodes: int = 15
    edge_prob: float = 0.5
    graph_type: str = "erdos_renyi"
    seed: Optional[int] = None


class Node(BaseModel):
    id: int
    degree: float
    clustering: float
    importance: float


class Edge(BaseModel):
    source: int
    target: int


class GraphData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    n_nodes: int
    n_edges: int
    density: float
    avg_clustering: float
    avg_degree: float


class PredictionResult(BaseModel):
    gamma: List[float]
    beta: List[float]
    p_layers: int
    graph: GraphData


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResult)
async def predict(request: GenerateRequest):
    """Generate a graph and predict QAOA parameters."""

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    seed = request.seed if request.seed else np.random.randint(10000)

    # Generate graph based on type
    try:
        if request.graph_type == "erdos_renyi":
            G = nx.erdos_renyi_graph(request.n_nodes, request.edge_prob, seed=seed)
        elif request.graph_type == "regular":
            degree = min(int(request.n_nodes * request.edge_prob * 2), request.n_nodes - 1)
            if degree < 2:
                degree = 2
            if (request.n_nodes * degree) % 2 != 0:
                degree = max(2, degree - 1)
            G = nx.random_regular_graph(degree, request.n_nodes, seed=seed)
        elif request.graph_type == "barabasi_albert":
            m = max(1, int(request.n_nodes * request.edge_prob))
            G = nx.barabasi_albert_graph(request.n_nodes, m, seed=seed)
        elif request.graph_type == "watts_strogatz":
            k = max(2, int(request.n_nodes * request.edge_prob * 2))
            if k % 2 != 0:
                k += 1
            G = nx.watts_strogatz_graph(request.n_nodes, k, 0.3, seed=seed)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown graph type: {request.graph_type}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graph generation failed: {str(e)}")

    # Check connectivity
    if not nx.is_connected(G):
        raise HTTPException(status_code=400, detail="Generated graph is disconnected. Try different parameters.")

    # Convert to PyG data
    data = graph_to_pyg_data(G)

    # Get node importance
    importance = get_node_importance(G, data)

    # Predict parameters
    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        output = model(batch)

    output = output.cpu().numpy()[0]
    gamma = output[:p_layers].tolist()
    beta = output[p_layers:].tolist()

    # Build node data
    degrees = dict(G.degree())
    clustering = nx.clustering(G)

    nodes = [
        Node(
            id=i,
            degree=degrees[i],
            clustering=clustering[i],
            importance=importance[i]
        )
        for i in range(G.number_of_nodes())
    ]

    edges = [Edge(source=u, target=v) for u, v in G.edges()]

    graph_data = GraphData(
        nodes=nodes,
        edges=edges,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        density=nx.density(G),
        avg_clustering=nx.average_clustering(G),
        avg_degree=sum(degrees.values()) / G.number_of_nodes()
    )

    return PredictionResult(
        gamma=gamma,
        beta=beta,
        p_layers=p_layers,
        graph=graph_data
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

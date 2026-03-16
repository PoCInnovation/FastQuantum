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
import time
import asyncio

from prototype_v1.model import QuantumGraphModel
from api.qiskit_utils import solve_classical_maxcut

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
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256,
        gnn_layers=4,
        transformer_layers=4,
        num_heads=8,
        num_problems=10,
        dropout=0.1
    ).to(device)

    # Note: Currently no trained weights available, using random weights.
    # To load weights later, add:
    # model.load_state_dict(torch.load('path_to_weights.pt', map_location=device))
    model.eval()

    print(f"QuantumGraphModel loaded on {device}")


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
    """Calculate node importance. For prototype_v1, we use static attributes like degree centrality instead of GAT attention."""
    # Since prototype_v1 doesn't easily expose individual attention weights per edge in the same format,
    # we return a normalized degree centrality for visualization purposes.
    # In a real scenario, this might come from the transformer attention or the probs output.
    n_nodes = G.number_of_nodes()
    centrality = nx.degree_centrality(G)
    importance = np.array([centrality[i] for i in range(n_nodes)])
    
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())
    else:
        importance = np.ones(n_nodes)
        
    return importance.tolist()


# Request/Response models
class GenerateRequest(BaseModel):
    n_nodes: int = 15
    edge_prob: float = 0.5
    graph_type: str = "erdos_renyi"
    problem_id: int = 0  # 0: MaxCut, 1: Vertex Cover, 2: Independent Set
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
    probs: List[float]
    predictions: List[int]
    graph: GraphData
    classical_predictions: Optional[List[int]] = None
    ai_execution_time: float
    classical_execution_time: Optional[float] = None
    speedup: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResult)
async def predict(request: GenerateRequest):
    """Generate a graph and predict QAOA parameters."""
    print(f"\n--- New Request: /predict ---")
    print(f"Parameters: {request.n_nodes} nodes, {request.edge_prob} edge prob, type: {request.graph_type}")

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
    
    print(f"Graph generated successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to PyG data
    data = graph_to_pyg_data(G)

    # Get node importance
    importance = get_node_importance(G, data)

    # Predict parameters (AI)
    print("Starting AI prediction...")
    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    start_ai = time.time()
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, problem_id=request.problem_id)
    
    # Extract prediction bits and probabilities
    probs = output['probs'][0].cpu().numpy().tolist()
    predictions = output['predictions'][0].cpu().numpy().tolist()
    
    ai_execution_time = time.time() - start_ai
    print(f"AI prediction finished in {ai_execution_time:.4f}s")
    print(f"AI predictions -> {predictions}")

    # Run Classical optimization (Greedy MaxCut)
    # We run this in a separate thread to avoid blocking the event loop
    print("Starting Classical Greedy optimization...")
    loop = asyncio.get_event_loop()
    try:
        c_predictions, c_time = await asyncio.wait_for(
            loop.run_in_executor(None, solve_classical_maxcut, G, seed),
            timeout=10.0
        )
        print(f"Classical optimization finished in {c_time:.4f}s")
        print(f"Classical params -> Predictions: {c_predictions}")
    except asyncio.TimeoutError:
        print("Classical execution timed out after 10s.")
        c_predictions, c_time = None, 0.0
    except Exception as e:
        print(f"Classical execution failed: {e}")
        c_predictions, c_time = None, 0.0

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

    speedup = (c_time / ai_execution_time) if ai_execution_time > 0 and c_time > 0 else 0.0

    return PredictionResult(
        probs=probs,
        predictions=predictions,
        graph=graph_data,
        classical_predictions=c_predictions,
        ai_execution_time=ai_execution_time,
        classical_execution_time=c_time,
        speedup=speedup
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

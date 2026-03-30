#!/usr/bin/env python3
"""
FastAPI server for QAOA GNN model inference.
Uses FastQuantumPredictor to predict warm-starting QAOA angles (gamma, beta),
then runs a warm-started Qiskit QAOA to produce the final solution.
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
import time
import asyncio

from fastquantum import FastQuantumPredictor
from api.qiskit_utils import solve_classical_maxcut, solve_qiskit_maxcut, solve_qiskit_maxcut_warmed

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

# Global predictor instance
predictor: Optional[FastQuantumPredictor] = None

# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "finetuned_maxcut_model.pt"


def load_model():
    """Load the trained model via FastQuantumPredictor."""
    global predictor

    checkpoint_path = str(DEFAULT_CHECKPOINT)
    predictor = FastQuantumPredictor(model_checkpoint=checkpoint_path)
    print(f"FastQuantumPredictor loaded from {checkpoint_path}")


def get_node_importance(G: nx.Graph) -> List[float]:
    """Calculate node importance using degree centrality for visualization."""
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
    problem_id: int = 0  # 0: MaxCut (only MaxCut supported with current checkpoint)
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
    qiskit_predictions: Optional[List[int]] = None
    ai_execution_time: float
    classical_execution_time: Optional[float] = None
    qiskit_execution_time: Optional[float] = None
    speedup: Optional[float] = None
    qiskit_speedup: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResult)
async def predict(request: GenerateRequest):
    """Generate a graph, predict QAOA warm-starting angles via FastQuantumPredictor,
    then run warm-started Qiskit QAOA for the final solution."""
    print(f"\n--- New Request: /predict ---")
    print(f"Parameters: {request.n_nodes} nodes, {request.edge_prob} edge prob, type: {request.graph_type}")

    if predictor is None:
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

    # Step 1: Predict warm-starting angles via FastQuantumPredictor (< 20ms)
    print("Starting AI prediction (warm-starting angles)...")
    start_ai = time.time()
    gamma, beta = predictor.predict(G, problem="MAXCUT")
    ai_time = time.time() - start_ai
    print(f"FastQuantumPredictor finished in {ai_time:.4f}s — gamma={gamma}, beta={beta}")

    # Step 2: Run warm-started Qiskit QAOA with predicted angles
    print("Running warm-started QAOA...")
    p_layers = predictor.p_layers
    ai_predictions, warmed_time, probs = solve_qiskit_maxcut_warmed(G, gamma, beta, p=p_layers)
    ai_execution_time = ai_time + warmed_time
    print(f"Warm-started QAOA finished in {warmed_time:.4f}s — predictions={ai_predictions}")

    # If probs is empty (graph had 0 nodes), fill with importance
    if not probs:
        probs = get_node_importance(G)

    # Run Classical and standard Qiskit QAOA in parallel for comparison
    print("Starting Classical and standard Qiskit optimizations...")
    loop = asyncio.get_event_loop()

    async def run_classical():
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, solve_classical_maxcut, G, seed),
                timeout=10.0
            )
        except Exception as e:
            print(f"Classical execution failed: {e}")
            return None, 0.0

    async def run_qiskit():
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, solve_qiskit_maxcut, G, 1, seed),
                timeout=30.0
            )
        except Exception as e:
            print(f"Qiskit QAOA execution failed: {e}")
            return None, 0.0

    (c_predictions, c_time), (q_predictions, q_time) = await asyncio.gather(run_classical(), run_qiskit())
    print(f"Classical finished in {c_time:.4f}s")
    print(f"Standard Qiskit QAOA finished in {q_time:.4f}s")

    # Build node data
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    importance = get_node_importance(G)

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
    q_speedup = (q_time / ai_execution_time) if ai_execution_time > 0 and q_time > 0 else 0.0

    return PredictionResult(
        probs=probs,
        predictions=ai_predictions,
        graph=graph_data,
        classical_predictions=c_predictions,
        qiskit_predictions=q_predictions,
        ai_execution_time=ai_execution_time,
        classical_execution_time=c_time,
        qiskit_execution_time=q_time,
        speedup=speedup,
        qiskit_speedup=q_speedup
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

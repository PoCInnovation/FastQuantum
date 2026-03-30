import os
import sys
import time
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Any

# Adjust paths to import our library and generation tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fastquantum as fq
from generate_v1_dataset import solve_brute_force, build_qaoa_circuit
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

app = FastAPI(title="FastQuantum Demo API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Serve frontend statically
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


class GraphRequest(BaseModel):
    n_nodes: int = 12
    p_edge: float = 0.5
    seed: int = 42

class CompareRequest(BaseModel):
    graph: Dict[str, Any]
    problem: str = "MAXCUT"


def dict_to_nx(graph_dict: dict) -> nx.Graph:
    """Helper to convert JSON graph back to NetworkX."""
    G = nx.Graph()
    G.add_nodes_from([n["id"] for n in graph_dict["nodes"]])
    for e in graph_dict["links"]:
        # force-graph mutates links in-place, replacing IDs with node objects
        u = e["source"].get("id", e["source"]) if isinstance(e["source"], dict) else e["source"]
        v = e["target"].get("id", e["target"]) if isinstance(e["target"], dict) else e["target"]
        G.add_edge(u, v)
    return G


@app.post("/api/generate_graph")
def generate_graph(req: GraphRequest):
    """Generates an Erdos-Renyi graph and returns it in a D3/Force-Graph friendly format."""
    G = nx.erdos_renyi_graph(req.n_nodes, req.p_edge, seed=req.seed)
    
    nodes = [{"id": i, "name": f"Qubit {i}"} for i in G.nodes()]
    links = [{"source": u, "target": v} for u, v in G.edges()]
    
    return {"nodes": nodes, "links": links, "num_edges": G.number_of_edges()}


@app.post("/api/compare_qaoa")
def compare_qaoa(req: CompareRequest):
    """
    Compares Standard QAOA Warm-Starting via COBYLA vs FastQuantum Predictor.
    Runs Qiskit COBYLA optimization and then runs the FastQuantum Predictor,
    evaluating their final energies and time taken.
    """
    try:
        G = dict_to_nx(req.graph)
        p = 1  # Standard p=1 QAOA depth
        
        # 1. Setup Qiskit Simulator and Circuit
        qc, gammas, betas, h = build_qaoa_circuit(G, p, req.problem)
        estimator = StatevectorEstimator()
        
        def cost_func(pv):
            res = estimator.run([(qc, h, {gammas[i]: pv[i] for i in range(p)} | {betas[i]: pv[p+i] for i in range(p)})]).result()
            return res[0].data.evs

        # 2. Qiskit COBYLA Baseline
        start_qiskit = time.time()
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        initial_point = [random.uniform(0, np.pi) for _ in range(p)] + [random.uniform(0, np.pi/2) for _ in range(p)]
        
        res = minimize(
            cost_func, 
            initial_point, 
            method='COBYLA', 
            options={'maxiter': 60}  # Limit iterations for web demo responsiveness
        )
        qiskit_time = time.time() - start_qiskit
        qiskit_energy = float(res.fun)
        qiskit_gamma = res.x[:p].tolist()
        qiskit_beta = res.x[p:].tolist()

        # 3. FastQuantum Predictor (IA)
        start_ia = time.time()
        model_paths_qaoa = [
            os.path.join(BASE_DIR, "best_qaoa_gat_model.pt"),
            os.path.join(BASE_DIR, "finetuned_maxcut_model.pt"),
            os.path.join(BASE_DIR, "mon_nouveau_modele.pt")
        ]
        model_p1 = next((p for p in model_paths_qaoa if os.path.exists(p)), model_paths_qaoa[0])
        
        predictor_angles = fq.FastQuantumPredictor(model_p1)
        ia_gamma, ia_beta = predictor_angles.predict(G, problem=req.problem)
        
        # Evaluate energy with IA angles
        ia_angles = ia_gamma + ia_beta
        ia_energy = float(cost_func(ia_angles))
        ia_time = time.time() - start_ia

        return {
            "qiskit": {
                "time_sec": qiskit_time,
                "energy": qiskit_energy,
                "gamma": qiskit_gamma,
                "beta": qiskit_beta
            },
            "ia": {
                "time_sec": ia_time,
                "energy": ia_energy,
                "gamma": ia_gamma,
                "beta": ia_beta
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare_solution")
def compare_solution(req: CompareRequest):
    """
    Compares the Exact Solution (Brute Force) vs FastQuantum Solution Predictor.
    Returns the final bitstrings, cut/objective values, and computation times.
    """
    try:
        G = dict_to_nx(req.graph)
        
        # 1. Exact Brute Force
        start_exact = time.time()
        exact_val, exact_sol = solve_brute_force(G, req.problem)
        exact_time = time.time() - start_exact

        # 2. FastQuantum Solution Predictor
        start_ia = time.time()
        model_p2 = os.path.join(BASE_DIR, "best_model_solar-sweep-1.pt")
        predictor_sol = fq.FastQuantumSolutionPredictor(model_p2)
        ia_sol = predictor_sol.predict(G, problem=req.problem)
        ia_time = time.time() - start_ia
        
        # Calculate IA objective value
        ia_val = 0
        if req.problem == "MAXCUT":
            for u, v in G.edges():
                if ia_sol[u] != ia_sol[v]: ia_val += 1
        elif req.problem == "MIS":
            is_independent = True
            for u, v in G.edges():
                if ia_sol[u] == 1 and ia_sol[v] == 1:
                    is_independent = False
            ia_val = sum(ia_sol) if is_independent else 0
        elif req.problem == "MAX_CLIQUE":
            is_clique = True
            selected = [node for node, val in enumerate(ia_sol) if val == 1]
            for idx1 in range(len(selected)):
                for idx2 in range(idx1+1, len(selected)):
                    if not G.has_edge(selected[idx1], selected[idx2]):
                        is_clique = False
            ia_val = sum(ia_sol) if is_clique else 0

        return {
            "exact": {
                "time_sec": exact_time,
                "value": exact_val,
                "solution": exact_sol
            },
            "ia": {
                "time_sec": ia_time,
                "value": ia_val,
                "solution": ia_sol
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

import warnings
from typing import List, Tuple
import time
import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def solve_qaoa_qiskit(G: nx.Graph, p_layers: int = 1, seed: int = 42) -> Tuple[List[float], List[float], float]:
    """
    Run QAOA on the given graph to find optimal parameters using Qiskit.
    Returns: (gamma, beta, execution_time)
    """
    start_time = time.time()
    
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return [], [], 0.0

    # Build MaxCut Hamiltonian: sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # To maximize cut, we minimize H = sum_{(u,v)} Z_u Z_v
    # Coeff is 1.0 for Z_u Z_v term.
    pauli_list = []
    for u, v in G.edges():
        pauli_str = ["I"] * n_nodes
        # Qiskit uses little-endian qubit ordering (q0 is rightmost)
        pauli_str[n_nodes - 1 - u] = "Z"
        pauli_str[n_nodes - 1 - v] = "Z" 
        pauli_list.append(("".join(pauli_str), 1.0))
    
    if not pauli_list:
         return [0.0]*p_layers, [0.0]*p_layers, time.time() - start_time
         
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    
    ansatz = QAOAAnsatz(hamiltonian, reps=p_layers)
    estimator = StatevectorEstimator()
    
    def cost_func(params):
        pub = (ansatz, [hamiltonian], [params])
        job = estimator.run([pub])
        result = job.result()[0]
        return float(result.data.evs[0])

    # Initial parameters
    initial_point = np.zeros(ansatz.num_parameters)
    
    try:
        res = minimize(cost_func, initial_point, method='COBYLA', options={'maxiter': 15})
        optimal_point = res.x
        
        gammas = []
        betas = []
        
        for param, value in zip(ansatz.parameters, optimal_point):
            if param.name.startswith("β"):
                betas.append(float(value))
            elif param.name.startswith("γ"):
                gammas.append(float(value))
                
        if len(betas) != p_layers or len(gammas) != p_layers:
            half = len(optimal_point) // 2
            betas = optimal_point[:half].tolist()
            gammas = optimal_point[half:].tolist()

    except Exception as e:
        print(f"QAOA Optimization failed: {e}")
        gammas = [0.0] * p_layers
        betas = [0.0] * p_layers
        
    execution_time = time.time() - start_time
    return gammas, betas, execution_time

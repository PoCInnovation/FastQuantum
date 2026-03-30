import time
from typing import List, Tuple
import numpy as np
import networkx as nx
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

def solve_classical_maxcut(G: nx.Graph, seed: int = 42) -> Tuple[List[int], float]:
    """
    Run a fast classical greedy algorithm to find an approximate MaxCut on the given graph.
    Returns: (best_bitstring_as_ints, execution_time)
    """
    start_time = time.time()
    n_nodes = G.number_of_nodes()

    if n_nodes == 0:
        return [], 0.0

    # Initialize a random partition
    np.random.seed(seed)
    partition = np.random.randint(0, 2, n_nodes)

    improved = True
    while improved:
        improved = False
        for i in range(n_nodes):
            # Calculate the gain of flipping the current node's partition
            # Gain = (edges to same partition) - (edges to other partition)
            gain = sum(1 if partition[i] == partition[neighbor] else -1 for neighbor in G.neighbors(i))

            if gain > 0:
                partition[i] = 1 - partition[i]
                improved = True

    execution_time = time.time() - start_time
    # ensure it takes at least 1ms to avoid division by zero
    if execution_time == 0:
        execution_time = 0.001

    return partition.tolist(), execution_time

def get_maxcut_hamiltonian(G: nx.Graph) -> Tuple[SparsePauliOp, float]:
    n = G.number_of_nodes()
    pauli_list = []
    for u, v in G.edges():
        op_str = ["I"] * n
        op_str[n - 1 - u] = "Z"
        op_str[n - 1 - v] = "Z"
        pauli_list.append(("".join(op_str), 1.0))
    if not pauli_list:
        return SparsePauliOp(["I" * n], [0.0]), 0.0
    return SparsePauliOp.from_list(pauli_list), 0.0

def build_qaoa_circuit(G: nx.Graph, p: int):
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    gammas = [Parameter(f'g{i}') for i in range(p)]
    betas = [Parameter(f'b{i}') for i in range(p)]
    h, _ = get_maxcut_hamiltonian(G)
    for layer in range(p):
        for op, coeff in zip(h.paulis, h.coeffs):
            z_idx = [k for k, c in enumerate(reversed(op.to_label())) if c == 'Z']
            if len(z_idx) == 1: qc.rz(2 * gammas[layer] * coeff.real, z_idx[0])
            elif len(z_idx) == 2: qc.rzz(2 * gammas[layer] * coeff.real, z_idx[0], z_idx[1])
        for i in range(n): qc.rx(2 * betas[layer], i)
    return qc, gammas, betas, h

def solve_qiskit_maxcut_warmed(G: nx.Graph, gamma: List[float], beta: List[float], p: int = 1) -> Tuple[List[int], float, List[float]]:
    """
    Run Qiskit QAOA warm-started with predicted gamma/beta angles from FastQuantumPredictor.
    Returns: (best_bitstring_as_ints, execution_time, per_node_marginal_probs)
    """
    start_time = time.time()
    n_nodes = G.number_of_nodes()

    if n_nodes == 0:
        return [], 0.001, []

    qc, gs, bs, h = build_qaoa_circuit(G, p)

    # Bind warm-started parameters directly (no optimization needed)
    bound_qc = qc.assign_parameters(
        {gs[i]: gamma[i] for i in range(p)} | {bs[i]: beta[i] for i in range(p)}
    )

    sv = Statevector(bound_qc)
    probs_dict = sv.probabilities_dict()
    best_bitstring = max(probs_dict, key=probs_dict.get)
    prediction = [int(b) for b in reversed(best_bitstring)]

    # Compute per-node marginal probability of being in partition 1
    marginal_probs = []
    for qubit in range(n_nodes):
        prob_1 = sum(
            p_val for bitstr, p_val in probs_dict.items()
            if bitstr[n_nodes - 1 - qubit] == '1'
        )
        marginal_probs.append(float(prob_1))

    execution_time = time.time() - start_time
    if execution_time == 0:
        execution_time = 0.001

    return prediction, execution_time, marginal_probs


def solve_qiskit_maxcut(G: nx.Graph, p: int = 1, seed: int = 42) -> Tuple[List[int], float]:
    """
    Run Qiskit QAOA wrapper to find an approximate MaxCut on the given graph.
    Returns: (best_bitstring_as_ints, execution_time)
    """
    start_time = time.time()
    n_nodes = G.number_of_nodes()

    if n_nodes == 0:
        return [], 0.001

    random.seed(seed)
    np.random.seed(seed)

    qc, gs, bs, h = build_qaoa_circuit(G, p)
    estimator = StatevectorEstimator()

    def cost(pv):
        res = estimator.run([(qc, h, {gs[i]: pv[i] for i in range(p)} | {bs[i]: pv[p+i] for i in range(p)})]).result()
        return res[0].data.evs

    best_e = float('inf')
    best_p = None

    for _ in range(2):
        res = minimize(cost, [random.uniform(0, np.pi) for _ in range(p)] + [random.uniform(0, np.pi/2) for _ in range(p)],
                       method='COBYLA', options={'maxiter': 50})
        if res.fun < best_e: best_e, best_p = res.fun, res.x

    bound_qc = qc.assign_parameters({gs[i]: best_p[i] for i in range(p)} | {bs[i]: best_p[p+i] for i in range(p)})

    sv = Statevector(bound_qc)
    probs = sv.probabilities_dict()
    best_bitstring = max(probs, key=probs.get)

    # Qiskit output is read qubit N-1 (left) to qubit 0 (right).
    # Reversing matches qubit 0 to index 0, qubit 1 to index 1...
    prediction = [int(b) for b in reversed(best_bitstring)]

    execution_time = time.time() - start_time
    if execution_time == 0:
        execution_time = 0.001

    return prediction, execution_time

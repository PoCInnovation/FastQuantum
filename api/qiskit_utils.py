import time
from typing import List, Tuple
import numpy as np
import networkx as nx

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

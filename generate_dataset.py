import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
import json
from pathlib import Path


class QAOADataGeneratorImproved:
    """
    Improved data generator for QAOA problems with more node features.
    Now includes 7 features instead of 3 for better GNN performance.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_erdos_renyi_graph(self, n_nodes: int, edge_prob: float) -> nx.Graph:
        """Generate an Erdős-Rényi random graph."""
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=self.seed)
        return G
    
    def generate_regular_graph(self, n_nodes: int, degree: int) -> nx.Graph:
        """Generate a regular graph where all nodes have the same degree."""
        if (n_nodes * degree) % 2 != 0:
            raise ValueError("n_nodes * degree must be even for regular graph")
        
        G = nx.random_regular_graph(degree, n_nodes, seed=self.seed)
        return G
    
    def generate_weighted_graph(self, n_nodes: int, edge_prob: float, 
                                weight_range: Tuple[float, float] = (0.1, 1.0)) -> nx.Graph:
        """Generate a weighted Erdős-Rényi graph."""
        G = self.generate_erdos_renyi_graph(n_nodes, edge_prob)
        
        for u, v in G.edges():
            weight = np.random.uniform(*weight_range)
            G[u][v]['weight'] = weight
        
        return G
    
    def graph_to_adjacency_matrix(self, G: nx.Graph) -> np.ndarray:
        """Convert graph to adjacency matrix."""
        return nx.to_numpy_array(G)
    
    def compute_node_features(self, G: nx.Graph) -> np.ndarray:
        """
        Compute 7 node features for better GNN performance:
        
        1. Degree - Number of neighbors
        2. Degree centrality - Normalized degree
        3. Clustering coefficient - How connected are neighbors
        4. Betweenness centrality - How many shortest paths pass through node
        5. Closeness centrality - Average distance to all other nodes
        6. PageRank - Importance based on connections (Google algorithm)
        7. Eigenvector centrality - Importance based on important neighbors
        
        Args:
            G: NetworkX graph
            
        Returns:
            Node feature matrix (n_nodes x 7)
        """
        n_nodes = G.number_of_nodes()
        features = np.zeros((n_nodes, 7))
        
        # Feature 0: Degree
        degrees = dict(G.degree())
        features[:, 0] = [degrees[i] for i in range(n_nodes)]
        
        # Feature 1: Degree centrality
        degree_centrality = nx.degree_centrality(G)
        features[:, 1] = [degree_centrality[i] for i in range(n_nodes)]
        
        # Feature 2: Clustering coefficient
        clustering = nx.clustering(G)
        features[:, 2] = [clustering[i] for i in range(n_nodes)]
        
        # Feature 3: Betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        features[:, 3] = [betweenness[i] for i in range(n_nodes)]
        
        # Feature 4: Closeness centrality
        closeness = nx.closeness_centrality(G)
        features[:, 4] = [closeness[i] for i in range(n_nodes)]
        
        # Feature 5: PageRank
        pagerank = nx.pagerank(G, max_iter=1000)
        features[:, 5] = [pagerank[i] for i in range(n_nodes)]
        
        # Feature 6: Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            features[:, 6] = [eigenvector[i] for i in range(n_nodes)]
        except:
            # If eigenvector centrality fails to converge, use zeros
            features[:, 6] = 0.0
        
        return features
    
    def graph_to_qubo(self, G: nx.Graph, hamiltonian_type: str = 'maxcut') -> Dict:
        """
        Convert graph to QUBO formulation for various optimization problems.
        
        Args:
            G: NetworkX graph
            hamiltonian_type: Type of optimization problem
                - 'maxcut': Maximum cut problem
                - 'mis': Maximum weight independent set
                - 'mvc': Minimum vertex cover
                - 'coloring': Graph coloring (simplified 2-coloring)
        
        Returns:
            Dictionary with 'h', 'J' arrays and 'hamiltonian_type'
        """
        n_nodes = G.number_of_nodes()
        adj_matrix = self.graph_to_adjacency_matrix(G)
        
        if hamiltonian_type == 'maxcut':
            # MaxCut: Maximize cut between two sets
            # QUBO: h_i = 0, J_ij = -weight(i,j)/2
            h = np.zeros(n_nodes)
            J = -0.5 * adj_matrix
            
        elif hamiltonian_type == 'mis':
            # Maximum Independent Set: Find largest set of non-adjacent nodes
            # We add node weights (degree-based) to make it more interesting
            # QUBO: h_i = -degree(i), J_ij = +large_penalty for adjacent nodes
            h = -np.array([len(list(G.neighbors(i))) for i in range(n_nodes)], dtype=float)
            J = np.zeros((n_nodes, n_nodes))
            penalty = 10.0  # Large penalty for selecting adjacent nodes
            for i, j in G.edges():
                J[i][j] = penalty
                J[j][i] = penalty
                
        elif hamiltonian_type == 'mvc':
            # Minimum Vertex Cover: Find smallest set covering all edges
            # QUBO: h_i = +1 (cost of selecting node), J_ij = -large_penalty for uncovered edges
            h = np.ones(n_nodes)
            J = np.zeros((n_nodes, n_nodes))
            penalty = -5.0  # Penalty for NOT covering an edge
            for i, j in G.edges():
                # If neither i nor j is selected, we get penalty
                # This translates to: (1-x_i)(1-x_j) * penalty = penalty - penalty*x_i - penalty*x_j + penalty*x_i*x_j
                h[i] -= penalty  # Linear term from expansion
                h[j] -= penalty
                J[i][j] += penalty  # Quadratic term
                
        elif hamiltonian_type == 'coloring':
            # Graph 2-coloring: Assign one of two colors such that adjacent nodes differ
            # Similar to MaxCut but with different interpretation
            # QUBO: h_i = 0, J_ij = +penalty for same color (adjacent nodes)
            h = np.zeros(n_nodes)
            J = np.zeros((n_nodes, n_nodes))
            penalty = 2.0
            for i, j in G.edges():
                J[i][j] = penalty
                J[j][i] = penalty
                
        else:
            raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")
        
        return {'h': h, 'J': J, 'hamiltonian_type': hamiltonian_type}
    
    def simulate_qaoa_optimization(self, G: nx.Graph, hamiltonian_type: str = 'maxcut', p: int = 1):
        """
        Simulate QAOA optimization for different Hamiltonian types.
        
        Different problems have different optimal parameter ranges:
        - MaxCut: Balanced gamma/beta
        - MIS: Higher gamma (more mixing)
        - MVC: Lower gamma (less mixing) 
        - Coloring: Similar to MaxCut
        """
        # Base parameters on graph structure
        avg_clustering = nx.average_clustering(G)
        
        gamma_values = []
        beta_values = []
        
        for layer in range(p):
            if hamiltonian_type == 'maxcut':
                # Original MaxCut parameters
                gamma = 0.35 + 0.40 * avg_clustering + np.random.normal(0, 0.03)
                gamma = np.clip(gamma, 0.25, 0.85)
                beta = 0.20 + 0.25 * (1 - avg_clustering) + np.random.normal(0, 0.02)
                beta = np.clip(beta, 0.15, 0.50)
                
            elif hamiltonian_type == 'mis':
                # MIS typically needs higher gamma for more exploration
                gamma = 0.50 + 0.30 * avg_clustering + np.random.normal(0, 0.04)
                gamma = np.clip(gamma, 0.40, 0.95)
                beta = 0.15 + 0.20 * (1 - avg_clustering) + np.random.normal(0, 0.02)
                beta = np.clip(beta, 0.10, 0.40)
                
            elif hamiltonian_type == 'mvc':
                # MVC typically needs lower gamma, higher beta
                gamma = 0.25 + 0.35 * avg_clustering + np.random.normal(0, 0.03)
                gamma = np.clip(gamma, 0.15, 0.70)
                beta = 0.30 + 0.25 * (1 - avg_clustering) + np.random.normal(0, 0.03)
                beta = np.clip(beta, 0.20, 0.60)
                
            elif hamiltonian_type == 'coloring':
                # Similar to MaxCut but slightly different ranges
                gamma = 0.40 + 0.35 * avg_clustering + np.random.normal(0, 0.03)
                gamma = np.clip(gamma, 0.30, 0.80)
                beta = 0.25 + 0.20 * (1 - avg_clustering) + np.random.normal(0, 0.02)
                beta = np.clip(beta, 0.18, 0.45)
            
            else:
                raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")
            
            gamma_values.append(gamma)
            beta_values.append(beta)
        
        return {'gamma': np.array(gamma_values), 'beta': np.array(beta_values)}
    
    def compute_approximation_ratio(self, G: nx.Graph, solution: np.ndarray, 
                                   hamiltonian_type: str = 'maxcut') -> float:
        """
        Compute approximation ratio for different optimization problems.
        
        Args:
            G: NetworkX graph
            solution: Binary solution vector
            hamiltonian_type: Type of problem
            
        Returns:
            Approximation ratio (actual_value / optimal_value)
        """
        n_nodes = len(solution)
        solution = solution.astype(int)
        
        if hamiltonian_type == 'maxcut':
            # Count edges in the cut
            cut_value = 0
            for u, v in G.edges():
                if solution[u] != solution[v]:
                    weight = G[u][v].get('weight', 1.0)
                    cut_value += weight
            max_cut = sum(G[u][v].get('weight', 1.0) for u, v in G.edges())
            approximation_ratio = cut_value / max_cut if max_cut > 0 else 0
            
        elif hamiltonian_type == 'mis':
            # Count selected nodes (weighted by degree), ensure independence
            independent_value = 0
            # Check if solution is actually independent
            for u, v in G.edges():
                if solution[u] == 1 and solution[v] == 1:
                    return 0.0  # Invalid solution
            
            for i in range(n_nodes):
                if solution[i] == 1:
                    independent_value += len(list(G.neighbors(i)))
            
            # Approximate maximum: sum of all degrees (upper bound)
            max_possible = sum(len(list(G.neighbors(i))) for i in range(n_nodes))
            approximation_ratio = independent_value / max_possible if max_possible > 0 else 0
            
        elif hamiltonian_type == 'mvc':
            # Count selected nodes, ensure all edges are covered
            cover_value = sum(solution)
            
            # Check if all edges are covered
            for u, v in G.edges():
                if solution[u] == 0 and solution[v] == 0:
                    return 0.0  # Invalid solution
            
            # For approximation ratio: smaller is better, so use 1 - (cover_size / n_nodes)
            approximation_ratio = 1.0 - (cover_value / n_nodes) if n_nodes > 0 else 0
            
        elif hamiltonian_type == 'coloring':
            # Similar to maxcut - count satisfied edges
            satisfied_edges = 0
            for u, v in G.edges():
                if solution[u] != solution[v]:
                    satisfied_edges += 1
            
            total_edges = G.number_of_edges()
            approximation_ratio = satisfied_edges / total_edges if total_edges > 0 else 1.0
            
        else:
            raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")
        
        return approximation_ratio
    
    def generate_dataset(self, n_samples: int, n_nodes_range: Tuple[int, int],
                        graph_type: str = 'erdos_renyi', hamiltonian_type: str = 'maxcut', 
                        p: int = 1, save_path: Optional[str] = None) -> List[Dict]:
        """
        Generate a complete dataset of graphs with QAOA parameters.
        Now supports multiple Hamiltonian types with 7 node features for better GNN performance!
        
        Args:
            n_samples: Number of graph samples to generate
            n_nodes_range: Range of number of nodes (min, max)
            graph_type: Type of graph ('erdos_renyi', 'regular', 'weighted')
            hamiltonian_type: Type of optimization problem ('maxcut', 'mis', 'mvc', 'coloring')
            p: Number of QAOA layers
            save_path: Path to save the dataset (optional)
            
        Returns:
            List of data samples
        """
        dataset = []
        
        for i in range(n_samples):
            n_nodes = np.random.randint(n_nodes_range[0], n_nodes_range[1] + 1)
            
            # Generate graph based on type
            if graph_type == 'erdos_renyi':
                edge_prob = np.random.uniform(0.3, 0.9)
                G = self.generate_erdos_renyi_graph(n_nodes, edge_prob)
            elif graph_type == 'regular':
                degree = min(np.random.randint(2, 5), n_nodes - 1)
                if (n_nodes * degree) % 2 != 0:
                    degree += 1
                G = self.generate_regular_graph(n_nodes, degree)
            elif graph_type == 'weighted':
                edge_prob = np.random.uniform(0.2, 0.9)
                G = self.generate_weighted_graph(n_nodes, edge_prob)
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
            
            # Skip disconnected graphs
            if not nx.is_connected(G):
                continue
            
            # Extract features (NOW 7 FEATURES!)
            adj_matrix = self.graph_to_adjacency_matrix(G)
            node_features = self.compute_node_features(G)
            qubo = self.graph_to_qubo(G, hamiltonian_type)
            
            # Get optimal parameters (simulated)
            optimal_params = self.simulate_qaoa_optimization(G, hamiltonian_type, p)
            
            # Create data sample
            sample = {
                'id': i,
                'n_nodes': n_nodes,
                'n_edges': G.number_of_edges(),
                'adjacency_matrix': adj_matrix.tolist(),
                'node_features': node_features.tolist(),
                'qubo_h': qubo['h'].tolist(),
                'qubo_J': qubo['J'].tolist(),
                'optimal_gamma': optimal_params['gamma'].tolist(),
                'optimal_beta': optimal_params['beta'].tolist(),
                'graph_type': graph_type,
                'hamiltonian_type': hamiltonian_type
            }
            
            dataset.append(sample)
            
            # Update seed for next iteration
            self.seed += 1
            np.random.seed(self.seed)
        
        # Save dataset if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Dataset saved to {save_path}")
        
        return dataset
    
    def load_dataset(self, load_path: str) -> List[Dict]:
        """Load a previously generated dataset."""
        with open(load_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Compute statistics on the dataset."""
        n_nodes = [sample['n_nodes'] for sample in dataset]
        n_edges = [sample['n_edges'] for sample in dataset]
        
        stats = {
            'n_samples': len(dataset),
            'avg_nodes': np.mean(n_nodes),
            'min_nodes': np.min(n_nodes),
            'max_nodes': np.max(n_nodes),
            'avg_edges': np.mean(n_edges),
            'min_edges': np.min(n_edges),
            'max_edges': np.max(n_edges),
            'n_features': 7  # Now 7 features!
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize improved generator
    generator = QAOADataGeneratorImproved(seed=42)
    
    # Demo: Generate datasets for different Hamiltonian types
    hamiltonian_types = ['maxcut', 'mis', 'mvc', 'coloring']
    
    print("Generating improved datasets with 7 node features for multiple Hamiltonian types...")
    print("Available Hamiltonian types:")
    print("  • MaxCut: Find maximum cut in graph")
    print("  • MIS: Maximum Independent Set")
    print("  • MVC: Minimum Vertex Cover") 
    print("  • Coloring: Graph 2-coloring")
    print()
    
    for hamiltonian_type in hamiltonian_types:
        print(f"Generating dataset for {hamiltonian_type.upper()}...")
        
        # Generate smaller training dataset for demo
        train_dataset = generator.generate_dataset(
            n_samples=500,  # Reduced for demo
            n_nodes_range=(8, 25),
            graph_type='erdos_renyi',
            hamiltonian_type=hamiltonian_type,
            p=1,
            save_path=f'Dataset/qaoa_{hamiltonian_type}_train_dataset.json'
        )
        
        # Generate validation dataset
        generator.seed += 1000
        val_dataset = generator.generate_dataset(
            n_samples=100,  # Reduced for demo
            n_nodes_range=(8, 25),
            graph_type='erdos_renyi',
            hamiltonian_type=hamiltonian_type,
            p=1,
            save_path=f'Dataset/qaoa_{hamiltonian_type}_val_dataset.json'
        )
        
        # Display statistics
        train_stats = generator.dataset_statistics(train_dataset)
        print(f"  Training: {train_stats['n_samples']} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print()
    
    print("All datasets generated successfully!")
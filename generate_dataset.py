import numpy as np
import networkx as nx
from typing import Tuple, List, Dict
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
    
    def graph_to_qubo(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """Convert graph to QUBO formulation for MaxCut problem."""
        n_nodes = G.number_of_nodes()
        adj_matrix = self.graph_to_adjacency_matrix(G)
        
        # For MaxCut: h_i = 0, J_ij = -weight(i,j)/2
        h = np.zeros(n_nodes)
        J = -0.5 * adj_matrix
        
        return {'h': h, 'J': J}
    
    def simulate_qaoa_optimization(self, G: nx.Graph, p: int = 1):
        # Base parameters on graph structure
        avg_clustering = nx.average_clustering(G)
        
        gamma_values = []
        beta_values = []
        
        for layer in range(p):
            # Gamma increases with clustering
            gamma = 0.35 + 0.40 * avg_clustering + np.random.normal(0, 0.03)
            gamma = np.clip(gamma, 0.25, 0.85)
            
            # Beta decreases with clustering  
            beta = 0.20 + 0.25 * (1 - avg_clustering) + np.random.normal(0, 0.02)
            beta = np.clip(beta, 0.15, 0.50)
            
            gamma_values.append(gamma)
            beta_values.append(beta)
        
        return {'gamma': np.array(gamma_values), 'beta': np.array(beta_values)}
    
    def compute_approximation_ratio(self, G: nx.Graph, solution: np.ndarray) -> float:
        """Compute approximation ratio for MaxCut problem."""
        cut_value = 0
        for u, v in G.edges():
            if solution[u] != solution[v]:
                weight = G[u][v].get('weight', 1.0)
                cut_value += weight
        
        max_cut = sum(G[u][v].get('weight', 1.0) for u, v in G.edges())
        approximation_ratio = cut_value / max_cut if max_cut > 0 else 0
        return approximation_ratio
    
    def generate_dataset(self, n_samples: int, n_nodes_range: Tuple[int, int],
                        graph_type: str = 'erdos_renyi', p: int = 1,
                        save_path: str = None) -> List[Dict]:
        """
        Generate a complete dataset of graphs with QAOA parameters.
        Now with 7 node features for better GNN performance!
        
        Args:
            n_samples: Number of graph samples to generate
            n_nodes_range: Range of number of nodes (min, max)
            graph_type: Type of graph ('erdos_renyi', 'regular', 'weighted')
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
            qubo = self.graph_to_qubo(G)
            
            # Get optimal parameters (simulated)
            optimal_params = self.simulate_qaoa_optimization(G, p)
            
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
                'graph_type': graph_type
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
    
    # Generate training dataset with 7 features
    print("Generating improved training dataset with 7 node features...")
    train_dataset = generator.generate_dataset(
        n_samples=3000,
        n_nodes_range=(8, 50),
        graph_type='erdos_renyi',
        p=1,
        save_path='Dataset/qaoa_train_dataset.json'
    )
    
    # Generate validation dataset
    print("\nGenerating improved validation dataset...")
    generator.seed = 1000
    val_dataset = generator.generate_dataset(
        n_samples=600,
        n_nodes_range=(8, 50),
        graph_type='erdos_renyi',
        p=1,
        save_path='Dataset/qaoa_val_dataset.json'
    )
    
    # Display statistics
    print("\nTraining dataset statistics:")
    train_stats = generator.dataset_statistics(train_dataset)
    for key, value in train_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nValidation dataset statistics:")
    val_stats = generator.dataset_statistics(val_dataset)
    for key, value in val_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n✅ Datasets generated with 7 features per node!")
    print("Features: degree, degree_centrality, clustering, betweenness,")
    print("          closeness, pagerank, eigenvector_centrality")
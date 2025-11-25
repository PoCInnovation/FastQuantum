"""
Test GNN Model with Real Qiskit QAOA Simulation

This script:
1. Loads the trained GNN model (best_qaoa_gat_model.pt)
2. Generates test graphs
3. Predicts gamma/beta with GNN
4. Evaluates the predictions using real Qiskit QAOA simulation
5. Compares GNN predictions with optimal QAOA parameters

Usage:
    python test_gnn_with_qiskit.py
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import sys
import os

# Add FastQuantum to path
sys.path.append('FastQuantum')
from GnnmodelGat import QAOAPredictorGAT

# Qiskit imports
try:
    from qiskit_algorithms.minimum_eigensolvers import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit.primitives import StatevectorSampler
    QISKIT_AVAILABLE = True
    print("[OK] Qiskit imported successfully")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"[ERROR] Qiskit not available: {e}")
    print("Install: pip install qiskit qiskit-algorithms qiskit-aer")


class GNNQiskitTester:
    """
    Test GNN predictions against real Qiskit QAOA simulations
    """

    def __init__(self, model_path='FastQuantum/best_qaoa_gat_model.pt'):
        """
        Initialize tester

        Args:
            model_path: Path to trained GNN model (.pt file)
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load GNN model
        print(f"\n[LOADING MODEL]")
        print(f"Model path: {model_path}")
        self.load_gnn_model()

        # Setup Qiskit simulator
        if QISKIT_AVAILABLE:
            print(f"\n[SETUP QISKIT]")
            self.sampler = StatevectorSampler()
            print(f"Sampler ready: StatevectorSampler")
        else:
            print("\n[WARNING] Qiskit not available - cannot run quantum simulations")

    def load_gnn_model(self):
        """Load trained GNN model from .pt file"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Check if checkpoint contains metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint format with metadata
                print(f"[INFO] Loading checkpoint with metadata:")
                print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"   Validation loss: {checkpoint.get('val_loss', 'unknown')}")
                print(f"   Model type: {checkpoint.get('model_type', 'unknown')}")

                p_layers = checkpoint.get('p_layers', 1)
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state dict
                p_layers = 1
                state_dict = checkpoint

            # Create model instance (must match training config)
            self.model = QAOAPredictorGAT(
                input_dim=7,      # Model trained with 7 features
                hidden_dim=64,
                num_layers=3,
                p_layers=p_layers,
                attention_heads=8,
                dropout=0.3
            )

            # Load trained weights
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[OK] Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   QAOA depth (p): {p_layers}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    def graph_to_pyg_data(self, G: nx.Graph) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data
        Using same 7 features as training:
        1. Degree
        2. Degree centrality
        3. Clustering coefficient
        4. Betweenness centrality
        5. Closeness centrality
        6. PageRank
        7. Eigenvector centrality

        Args:
            G: NetworkX graph

        Returns:
            PyTorch Geometric Data object
        """
        n_nodes = G.number_of_nodes()

        # Node features (7 features like training data)
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

        # Edge index
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        # Make undirected (add reverse edges)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Create PyG Data
        x = torch.tensor(features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        return data

    def predict_with_gnn(self, G: nx.Graph):
        """
        Predict QAOA parameters using GNN

        Args:
            G: NetworkX graph

        Returns:
            gamma, beta (numpy arrays)
        """
        # Convert to PyG data
        data = self.graph_to_pyg_data(G)
        data = data.to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(data)

        # Extract gamma and beta
        # Output format: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        output = output.cpu().numpy()[0]
        p = len(output) // 2
        gamma = output[:p]
        beta = output[p:]

        return gamma, beta

    def graph_to_maxcut_hamiltonian(self, G: nx.Graph) -> SparsePauliOp:
        """
        Convert graph to MaxCut Hamiltonian

        H = 0.5 * sum_{(i,j) in E} Z_i Z_j
        """
        n_nodes = G.number_of_nodes()
        pauli_list = []
        coeffs = []

        for (i, j) in G.edges():
            pauli_str = ['I'] * n_nodes
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(0.5)

        if len(pauli_list) == 0:
            pauli_list = ['I' * n_nodes]
            coeffs = [0.0]

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    def evaluate_with_qiskit(self, G: nx.Graph, gamma, beta):
        """
        Fonction avec evaluation final en simulant un vrais cicuitt puis en trouvant l'energie

        Args:
            G: NetworkX graph
            gamma: QAOA gamma parameters
            beta: QAOA beta parameters

        Returns:
            energy: Expected energy from QAOA

        
        """
        if not QISKIT_AVAILABLE:
            print("[ERROR] Qiskit not available")
            return None

        # Create Hamiltonian
        hamiltonian = self.graph_to_maxcut_hamiltonian(G)

        # Create QAOA with FIXED parameters (no optimization)
        # We want to evaluate the GNN's parameters, not optimize
        initial_point = np.concatenate([gamma, beta])

        # Use COBYLA with maxiter=1 to just evaluate, not optimize
        optimizer = COBYLA(maxiter=1)

        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=optimizer,
            reps=len(gamma),
            initial_point=initial_point
        )

        # Run QAOA (will just evaluate initial_point)
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        energy = result.optimal_value

        ## Ce qu'on pourrait faire (complexe):
        #quantum_state = simulate_full_circuit(G, gamma, beta)
        #bitstring = measure_state(quantum_state)  # Ex: "01011"
        #solution = decode_bitstring_to_partition(bitstring)
        #cut_edges = count_cut_edges(solution, G)
        #approx_ratio = cut_edges / max_possible_cuts

        return energy

    def find_optimal_with_qiskit(self, G: nx.Graph, p=1):
        """
        Find optimal QAOA parameters using Qiskit optimization

        Args:
            G: NetworkX graph
            p: QAOA depth

        Returns:
            gamma_opt, beta_opt, energy_opt
        """
        if not QISKIT_AVAILABLE:
            print("[ERROR] Qiskit not available")
            return None, None, None

        print(f"   [Running QAOA optimization...]", end=" ", flush=True)

        hamiltonian = self.graph_to_maxcut_hamiltonian(G)

        # Use SPSA for faster optimization with REDUCED iterations
        optimizer = SPSA(maxiter=10)  # Reduced from 30 to 10 for speed

        # Smart initialization (similar to GNN prediction) //optimisation d'optimisation avec donné "normalemnt corecte"
        avg_clustering = nx.average_clustering(G)
        gamma_init = 0.4 + 0.3 * avg_clustering
        beta_init = 0.3
        initial_point = np.array([gamma_init, beta_init])

        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=optimizer,
            reps=p,
            initial_point=initial_point  # Start closer to solution
        )

        result = qaoa.compute_minimum_eigenvalue(hamiltonian)

        optimal_point = result.optimal_point
        gamma_opt = optimal_point[:p]
        beta_opt = optimal_point[p:2*p]
        energy_opt = result.optimal_value

        print(f"Done")

        return gamma_opt, beta_opt, energy_opt

    def compute_approximation_ratio(self, G: nx.Graph, energy: float) -> float:
        """
        Compute approximation ratio for MaxCut

        approximation_ratio = achieved_cut / max_possible_cut
        """
        # Max possible cut (all edges cut)
        max_cut = G.number_of_edges()

        # Convert energy to cut value
        # Energy = 0.5 * sum(Z_i Z_j) ranges from -max_cut/2 to +max_cut/2
        # Cut value = (max_cut/2 - energy)
        cut_value = (max_cut / 2.0 - energy)

        approximation_ratio = cut_value / max_cut if max_cut > 0 else 0

        return approximation_ratio

    def test_single_graph(self, G: nx.Graph, graph_id=0):
        """
        Test GNN on a single graph

        Args:
            G: NetworkX graph
            graph_id: Graph identifier for display
        """
        print(f"\n{'='*70}")
        print(f"GRAPH {graph_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"{'='*70}")

        # 1. Predict with GNN
        print(f"\n1. GNN Prediction:")
        gamma_gnn, beta_gnn = self.predict_with_gnn(G)
        print(f"   gamma: {gamma_gnn}")
        print(f"   beta:  {beta_gnn}")

        if not QISKIT_AVAILABLE:
            print("\n[SKIP] Qiskit evaluation (not available)")
            return

        # 2. Evaluate GNN prediction with Qiskit
        print(f"\n2. Evaluating GNN prediction with Qiskit:")
        energy_gnn = self.evaluate_with_qiskit(G, gamma_gnn, beta_gnn)
        approx_ratio_gnn = self.compute_approximation_ratio(G, energy_gnn)
        print(f"   Energy: {energy_gnn:.4f}") #les deux data final
        print(f"   Approximation ratio: {approx_ratio_gnn:.2%}")

        # 3. Find optimal with Qiskit
        print(f"\n3. Finding optimal with Qiskit QAOA:")
        gamma_opt, beta_opt, energy_opt = self.find_optimal_with_qiskit(G, p=len(gamma_gnn))
        approx_ratio_opt = self.compute_approximation_ratio(G, energy_opt)
        print(f"   gamma: {gamma_opt}")
        print(f"   beta:  {beta_opt}")
        print(f"   Energy: {energy_opt:.4f}")
        print(f"   Approximation ratio: {approx_ratio_opt:.2%}")

        # 4. Comparison
        print(f"\n4. Comparison:")
        energy_gap = energy_gnn - energy_opt
        ratio_gap = approx_ratio_opt - approx_ratio_gnn

        print(f"   Energy gap: {energy_gap:.4f} (lower is better)")
        print(f"   Approx ratio gap: {ratio_gap:.2%} (lower is better)")

        # Parameter differences
        gamma_diff = np.linalg.norm(gamma_gnn - gamma_opt)
        beta_diff = np.linalg.norm(beta_gnn - beta_opt)
        print(f"   Gamma L2 distance: {gamma_diff:.4f}")
        print(f"   Beta L2 distance: {beta_diff:.4f}")

        # Verdict
        print(f"\n5. Verdict:")
        if ratio_gap < 0.05:  # Within 5%
            print(f"   [EXCELLENT] GNN prediction is very close to optimal!")
        elif ratio_gap < 0.10:  # Within 10%
            print(f"   [GOOD] GNN prediction is reasonably close to optimal")
        elif ratio_gap < 0.20:  # Within 20%
            print(f"   [ACCEPTABLE] GNN prediction is acceptable")
        else:
            print(f"   [POOR] GNN prediction needs improvement")

        return {
            'gamma_gnn': gamma_gnn,
            'beta_gnn': beta_gnn,
            'energy_gnn': energy_gnn,
            'approx_ratio_gnn': approx_ratio_gnn,
            'gamma_opt': gamma_opt,
            'beta_opt': beta_opt,
            'energy_opt': energy_opt,
            'approx_ratio_opt': approx_ratio_opt,
            'energy_gap': energy_gap,
            'ratio_gap': ratio_gap
        }

    def test_multiple_graphs(self, n_graphs=5, n_nodes_range=(8, 15)):
        """
        Test GNN on multiple random graphs

        Args:
            n_graphs: Number of graphs to test
            n_nodes_range: Range of number of nodes
        """
        print(f"\n{'#'*70}")
        print(f"# TESTING GNN WITH QISKIT ON {n_graphs} RANDOM GRAPHS")
        print(f"{'#'*70}")

        results = []

        for i in range(n_graphs):
            # Generate random graph
            n_nodes = np.random.randint(n_nodes_range[0], n_nodes_range[1] + 1)
            edge_prob = np.random.uniform(0.3, 0.7)
            G = nx.erdos_renyi_graph(n_nodes, edge_prob)

            # Skip disconnected graphs
            if not nx.is_connected(G):
                print(f"\n[SKIP] Graph {i} is disconnected")
                continue

            # Test
            result = self.test_single_graph(G, graph_id=i)
            if result:
                results.append(result)

        # Summary statistics
        if results and QISKIT_AVAILABLE:
            print(f"\n{'='*70}")
            print(f"SUMMARY STATISTICS ({len(results)} graphs)")
            print(f"{'='*70}")

            avg_energy_gap = np.mean([r['energy_gap'] for r in results])
            avg_ratio_gap = np.mean([r['ratio_gap'] for r in results])

            print(f"Average energy gap: {avg_energy_gap:.4f}")
            print(f"Average approximation ratio gap: {avg_ratio_gap:.2%}")

            # Best and worst
            best_idx = np.argmin([r['ratio_gap'] for r in results])
            worst_idx = np.argmax([r['ratio_gap'] for r in results])

            print(f"\nBest performance: Graph {best_idx} (gap: {results[best_idx]['ratio_gap']:.2%})")
            print(f"Worst performance: Graph {worst_idx} (gap: {results[worst_idx]['ratio_gap']:.2%})")

            # Overall verdict
            if avg_ratio_gap < 0.05:
                print(f"\n[OVERALL VERDICT] EXCELLENT - Model generalizes very well!")
            elif avg_ratio_gap < 0.10:
                print(f"\n[OVERALL VERDICT] GOOD - Model performs well")
            elif avg_ratio_gap < 0.20:
                print(f"\n[OVERALL VERDICT] ACCEPTABLE - Model is usable")
            else:
                print(f"\n[OVERALL VERDICT] NEEDS IMPROVEMENT")


def main():
    """Main test function"""
    print("="*70)
    print("GNN + QISKIT INTEGRATION TEST")
    print("="*70)

    # Initialize tester
    tester = GNNQiskitTester(model_path='FastQuantum/best_qaoa_gat_model.pt')

    # Test on multiple graphs (reduced to 3 for speed)
    tester.test_multiple_graphs(n_graphs=3, n_nodes_range=(8, 10))

    print(f"\n{'='*70}")
    print("TEST COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

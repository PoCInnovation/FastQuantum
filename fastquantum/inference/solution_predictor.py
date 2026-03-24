"""
FastQuantumSolutionPredictor — High-level API for direct solution bitstring prediction.

Takes a NetworkX graph and returns the optimal solution as a binary list [0, 1, 0, 1, ...],
where 1 means the node is selected (e.g., belongs to the MaxCut partition A, MIS set, etc.).
"""

import torch
import networkx as nx
from typing import List, Optional

from ..features.extractors import preprocess_graph
from ..models.quantum_graph_model import QuantumGraphModel

# Default problem name → problem_id mapping (must match training)
DEFAULT_PROBLEM_MAP = {
    "MAXCUT": 0,
    "MIS": 2,        # IndependentSet in the registry
    "MAX_CLIQUE": 3,
}


class FastQuantumSolutionPredictor:
    """
    High-level API for predicting optimal solution bitstrings using a pre-trained
    QuantumGraphModel (GAT Encoder + Graph Transformer + Symmetric BCE Classifier).

    The model predicts directly the binary partition of nodes that optimizes the
    objective function (e.g., maximizes the cut, the independent set, etc.).
    """

    def __init__(self, model_checkpoint: str, device: Optional[str] = None):
        """
        Initialize the solution predictor with a pre-trained model.

        Args:
            model_checkpoint (str): Path to the saved model checkpoint (.pt file).
            device (str, optional): 'cpu', 'cuda', etc. Auto-detects if None.
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model_checkpoint = model_checkpoint
        self.problem_map = DEFAULT_PROBLEM_MAP
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads the QuantumGraphModel from checkpoint, auto-inferring hyperparams."""
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)

        config = checkpoint.get('config', {})
        sd = checkpoint.get('model_state_dict', checkpoint)

        # Infer input dimension from first linear layer
        node_input_dim = config.get('node_input_dim',
            sd['encoder.gat_layers.0.lin.weight'].shape[1])

        # att_src shape: [1, num_heads_encoder, head_dim]
        # embedding_dim = num_heads_encoder * head_dim
        att_shape = sd['encoder.gat_layers.0.att_src'].shape
        num_heads_encoder = int(att_shape[1])
        embedding_dim = int(att_shape[1] * att_shape[2])

        gnn_layers = config.get('gnn_layers',
            sum(1 for k in sd if 'encoder.gat_layers' in k and 'att_src' in k))
        transformer_layers = config.get('transformer_layers',
            sum(1 for k in sd if 'transformer.transformer.layers.' in k and 'norm1.weight' in k))
        num_problems = config.get('num_problems',
            int(sd['problem_embedding.embedding_table.weight'].shape[0]))

        # QuantumGraphModel passes num_heads // 2 to GNNEncoder internally,
        # so we pass num_heads_encoder * 2 to match the checkpoint's encoder value.
        self.model = QuantumGraphModel(
            node_input_dim=node_input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            gnn_layers=gnn_layers,
            transformer_layers=transformer_layers,
            num_heads=num_heads_encoder * 2,
            num_problems=num_problems,
            dropout=0.0
        )

        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Loaded QuantumGraphModel | input_dim={node_input_dim} | "
              f"embedding={embedding_dim} | gnn_layers={gnn_layers} | "
              f"transformer_layers={transformer_layers} | device={self.device}")

    def predict(self, G: nx.Graph, problem: str = "MAXCUT") -> List[int]:
        """
        Predict the optimal solution bitstring for a given graph.

        Args:
            G (nx.Graph): The input graph (nodes = qubits, edges = interactions).
            problem (str): The optimization problem type. Supported: 'MAXCUT', 'MIS', 'MAX_CLIQUE'.

        Returns:
            List[int]: Binary solution bitstring of length n_nodes.
                       E.g. [0, 1, 0, 1, 1, 0] where 1 = node selected.
        """
        if problem not in self.problem_map:
            raise ValueError(
                f"Unknown problem type '{problem}'. Supported: {list(self.problem_map.keys())}"
            )

        problem_id = self.problem_map[problem]

        # 1. Auto feature extraction (7 heuristics + 16 RWPE = 23 features per node)
        G = nx.convert_node_labels_to_integers(G)
        pyg_data = preprocess_graph(G, rwpe_steps=16)
        pyg_data = pyg_data.to(self.device)

        # 2. Model inference
        with torch.no_grad():
            output = self.model(
                x=pyg_data.x,
                edge_index=pyg_data.edge_index,
                problem_id=problem_id,
                batch=None
            )

        # 3. Return the binary prediction [n_nodes]
        predictions = output['predictions'].squeeze(0).cpu().tolist()
        return predictions

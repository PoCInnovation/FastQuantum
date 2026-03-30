import torch
import networkx as nx
from typing import Tuple, Optional

from ..features.extractors import preprocess_graph, compute_node_features
from ..models.gat import QAOAPredictorGAT
from ..models.gcn import QAOAPredictorGCN


class FastQuantumPredictor:
    """
    High-level API for predicting optimal QAOA parameters directly from a
    NetworkX graph using a pre-trained Graph Neural Network.

    This class handles automatic feature extraction, data formatting,
    and model inference. It acts as the main entry point for the library.
    """

    def __init__(self, model_checkpoint: str, device: Optional[str] = None):
        """
        Initialize the predictor with a pre-trained model.

        Args:
            model_checkpoint (str): Path to the saved model checkpoint (.pt file)
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.).
                                    If None, automatically selects cuda if available.
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.p_layers = 1
        self.problem_map = {}
        self._input_dim = 7  # detected from checkpoint

        self._load_model()

    def _load_model(self):
        """Loads the model architecture and weights from the checkpoint, auto-inferring hyperparams."""
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)

        self.p_layers = checkpoint.get('p_layers', 1)
        model_type = checkpoint.get('model_type', 'GAT')
        sd = checkpoint.get('model_state_dict', checkpoint)

        if model_type == 'GAT':
            # Auto-detect architecture from state dict
            input_dim = sd['convs.0.lin.weight'].shape[1]
            hidden_dim = sd['batch_norms.0.weight'].shape[0]
            num_layers = sum(1 for k in sd if 'att_src' in k)
            use_edge_features = 'convs.0.att_edge' in sd
            use_input_bn = 'input_bn.weight' in sd

            self._input_dim = input_dim

            self.model = QAOAPredictorGAT(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                p_layers=self.p_layers,
                attention_heads=8,
                dropout=0.0,
                use_edge_features=use_edge_features,
                use_input_bn=use_input_bn,
            )
        else:
            input_dim = checkpoint.get('input_dim', 7)
            self._input_dim = input_dim
            self.model = QAOAPredictorGCN(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=3,
                p_layers=self.p_layers,
                dropout=0.0
            )

        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        # Load problem mapping if it exists in checkpoint, otherwise use a default
        self.problem_map = checkpoint.get('problem_map', {'MAXCUT': 0, 'MIS': 1, 'MAX_CLIQUE': 2})
        self.num_problems = len(self.problem_map)

        print(f"✅ Loaded QAOAPredictorGAT | input_dim={self._input_dim} | "
              f"hidden_dim={hidden_dim if model_type == 'GAT' else 64} | "
              f"num_layers={num_layers if model_type == 'GAT' else 3} | device={self.device}")

    def predict(self, G: nx.Graph, problem: str = "MAXCUT") -> Tuple[list, list]:
        """
        Predict optimal gamma and beta parameters for a given graph.

        Args:
            G (nx.Graph): The input graph representing the problem.
            problem (str): The combinatorial problem type (default: "MAXCUT").
                           Must be one of the problems the model was trained on.

        Returns:
            Tuple[list, list]: A tuple containing two lists: (gamma, beta).
                               gamma controls the cost Hamiltonian phase separator.
                               beta controls the mixing Hamiltonian.
        """
        if problem not in self.problem_map:
            raise ValueError(f"Unknown problem type '{problem}'. Supported types: {list(self.problem_map.keys())}")

        G = nx.convert_node_labels_to_integers(G)

        # Build features matching the model's input_dim
        # input_dim=7  → 7 heuristic features only
        # input_dim=23 → 7 heuristics + 16 RWPE
        # input_dim=26 → 7 heuristics + 16 RWPE + 3 one-hot
        base_features = 7
        rwpe_steps = 16

        if self._input_dim == base_features:
            # Legacy checkpoint: heuristics only, no RWPE, no one-hot
            node_feats = compute_node_features(G)
            x = torch.tensor(node_feats, dtype=torch.float32)
        elif self._input_dim == base_features + rwpe_steps:
            # heuristics + RWPE
            pyg_data = preprocess_graph(G, rwpe_steps=rwpe_steps)
            x = pyg_data.x
        else:
            # heuristics + RWPE + one-hot problem embedding
            pyg_data = preprocess_graph(G, rwpe_steps=rwpe_steps)
            prob_idx = self.problem_map[problem]
            one_hot = torch.zeros(self.num_problems)
            one_hot[prob_idx] = 1.0
            x = torch.cat([pyg_data.x, one_hot.repeat(pyg_data.x.size(0), 1)], dim=1)

        # Build PyG data object
        from torch_geometric.data import Data
        from torch_geometric.utils import from_networkx
        pyg_data = from_networkx(G)
        pyg_data.x = x
        pyg_data.batch = torch.zeros(x.size(0), dtype=torch.long)
        pyg_data = pyg_data.to(self.device)

        with torch.no_grad():
            output = self.model(pyg_data)

        output = output.cpu().numpy().flatten()
        gamma = output[:self.p_layers].tolist()
        beta = output[self.p_layers:].tolist()

        return gamma, beta

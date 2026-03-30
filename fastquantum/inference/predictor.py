import torch
import networkx as nx
from typing import Tuple, Dict, Optional, Union

from ..features.extractors import preprocess_graph
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
        
        self._load_model()
        
    def _load_model(self):
        """Loads the model architecture and weights from the checkpoint."""
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
        
        self.p_layers = checkpoint.get('p_layers', 1)
        model_type = checkpoint.get('model_type', 'GAT')
        
        # Inference requires knowing the input dimension, which depends on features + one_hot length.
        # Ideally, this should be saved in the checkpoint. 
        # For backward compatibility with older checkpoints, we deduce it or use a default.
        input_dim = checkpoint.get('input_dim', None)
        
        if model_type == 'GAT':
            # Default input_dim if not in checkpoint: 7 (heuristics) + 16 (RWPE) + 3 (num_problems) = 26
            input_dim = input_dim if input_dim is not None else 26
            
            # Auto-infer hidden_dim if possible
            sd = checkpoint['model_state_dict']
            hidden_dim = 128
            if 'gc1.weight' in sd:
                hidden_dim = sd['gc1.weight'].shape[0]
            
            self.model = QAOAPredictorGAT(
                input_dim=input_dim,
                hidden_dim=hidden_dim,      
                num_layers=4,        
                p_layers=self.p_layers,
                attention_heads=8,
                dropout=0.0 # Inference mode doesn't need dropout
            )
        else:
            input_dim = input_dim if input_dim is not None else 7
            self.model = QAOAPredictorGCN(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=3,
                p_layers=self.p_layers,
                dropout=0.0
            )
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load problem mapping if it exists in checkpoint, otherwise use a default
        self.problem_map = checkpoint.get('problem_map', {'MAXCUT': 0, 'MIS': 1, 'MAX_CLIQUE': 2})
        self.num_problems = len(self.problem_map)

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
            
        # 1. Automatic Feature Extraction
        # Default RWPE steps to 16, which is the library standard
        pyg_data = preprocess_graph(G, rwpe_steps=16)
        
        # 2. Add Problem Embedding (One-Hot)
        prob_idx = self.problem_map[problem]
        one_hot = torch.zeros(self.num_problems)
        one_hot[prob_idx] = 1.0
        expanded_one_hot = one_hot.repeat(pyg_data.x.size(0), 1)
        
        # Concat [Features, One-Hot]
        pyg_data.x = torch.cat([pyg_data.x, expanded_one_hot], dim=1)
        
        # Add dummy batch attribute for PyTorch Geometric pooling layers
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long)
        
        pyg_data = pyg_data.to(self.device)
        
        # 3. Model Inference
        with torch.no_grad():
            output = self.model(pyg_data)
        
        # 4. Format Output
        output = output.cpu().numpy().flatten()
        gamma = output[:self.p_layers].tolist()
        beta = output[self.p_layers:].tolist()
        
        return gamma, beta

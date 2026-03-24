"""
Problem Embedding Table for FastQuantum library.

Maps an integer problem ID to a learnable dense embedding vector.
The table is trained end-to-end during model training.
"""

import torch
import torch.nn as nn

# Default problem registry used during training
PROBLEM_REGISTRY = {
    0: "MaxCut",
    1: "VertexCover",
    2: "IndependentSet",
    3: "GraphColoring",
}


class ProblemEmbeddingTable(nn.Module):
    """
    Lookup table: problem_id → dense embedding vector.
    Learned via backpropagation.
    """

    def __init__(self, num_problems=10, embedding_dim=1024):
        super().__init__()
        self.num_problems = num_problems
        self.embedding_dim = embedding_dim
        self.embedding_table = nn.Embedding(num_problems, embedding_dim)
        nn.init.normal_(self.embedding_table.weight, mean=0.0, std=0.1)

    def forward(self, problem_id):
        """
        Args:
            problem_id: int or Tensor [batch_size]
        Returns:
            embedding: [batch_size, embedding_dim]
        """
        if isinstance(problem_id, int):
            problem_id = torch.tensor([problem_id])
        if not isinstance(problem_id, torch.Tensor):
            problem_id = torch.tensor(problem_id)
        problem_id = problem_id.to(self.embedding_table.weight.device)
        return self.embedding_table(problem_id)

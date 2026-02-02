"""
Table d'Embedding pour les Problem IDs

ID (ex: 2) → Lookup Table → Vector (1, 128)

Table:
(0, [0.654, 0.352, ..., 0.374])  MaxCut
(1, [0.376, 0.023, ..., 0.332])  VertexCover
(2, [0.372, 0.832, ..., 0.374])  IndependentSet
(3, [0.103, 0.334, ..., 0.743])  GraphColoring
"""

import torch
import torch.nn as nn


PROBLEM_REGISTRY = {
    0: "MaxCut",
    1: "VertexCover",
    2: "IndependentSet",
    3: "GraphColoring",
}


class ProblemEmbeddingTable(nn.Module):
    """
    Table d'embedding: problem_id → vecteur dense.
    Appris par backpropagation.
    """

    def __init__(self, num_problems=10, embedding_dim=128):
        super().__init__()
        self.num_problems = num_problems
        self.embedding_dim = embedding_dim

        # Lookup table
        self.embedding_table = nn.Embedding(num_problems, embedding_dim)
        nn.init.normal_(self.embedding_table.weight, mean=0.0, std=0.1)

    def forward(self, problem_id):
        """
        Args:
            problem_id: int ou Tensor [batch_size]
        Returns:
            embedding: [batch_size, embedding_dim]
        """
        if isinstance(problem_id, int):
            problem_id = torch.tensor([problem_id])
        if not isinstance(problem_id, torch.Tensor):
            problem_id = torch.tensor(problem_id)

        problem_id = problem_id.to(self.embedding_table.weight.device)
        return self.embedding_table(problem_id)


if __name__ == "__main__":
    table = ProblemEmbeddingTable(num_problems=10, embedding_dim=128)
    emb = table(2)
    print(f"Problem ID 2 → shape: {emb.shape}")
    print(f"Params: {sum(p.numel() for p in table.parameters()):,}")

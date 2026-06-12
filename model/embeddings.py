import torch
import torch.nn as nn
import numpy as np


class Embeddings(nn.Module):

    def __init__(
        self,
        embedding_path="embedding_matrix.npy",
        position_path="position_embeddings.npy"
    ):
        super().__init__()
        
        import os
        # Resolve paths relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        embedding_path = os.path.join(project_root, embedding_path)
        position_path = os.path.join(project_root, position_path)

        embedding_matrix = np.load(embedding_path)
        position_matrix = np.load(position_path)

        self.token_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            ),
            freeze=False
        )

        self.position_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(
                position_matrix,
                dtype=torch.float32
            ),
            freeze=False
        )

    def forward(self, token_ids):

        batch_size, seq_len = token_ids.shape

        positions = torch.arange(
            seq_len,
            device=token_ids.device
        ).unsqueeze(0)

        token_vectors = self.token_embeddings(
            token_ids
        )

        position_vectors = self.position_embeddings(
            positions
        )

        return token_vectors + position_vectors
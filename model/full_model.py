import torch
import torch.nn as nn

from .embeddings import Embeddings
from .transformer import Transformer
from .mlm_head import MLMHead


class FullModel(nn.Module):

    def __init__(
        self,
        vocab_size=10000
    ):
        super().__init__()

        self.embeddings = Embeddings(
            embedding_path="embedding_matrix.npy",
            position_path="position_embeddings.npy"
        )

        self.transformer = Transformer(
            num_layers=6,
            embed_dim=384,
            num_heads=12,
            hidden_dim=1536
        )

        self.mlm_head = MLMHead(
            embed_dim=384,
            vocab_size=vocab_size
        )

    def forward(
        self,
        input_ids
    ):

        x = self.embeddings(
            input_ids
        )

        x = self.transformer(
            x
        )

        logits = self.mlm_head(
            x
        )

        return logits
import torch
import torch.nn as nn

from .encoder import EncoderBlock


class Transformer(nn.Module):

    def __init__(
        self,
        num_layers=6,
        embed_dim=384,
        num_heads=12,
        hidden_dim=1536
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim,
                num_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
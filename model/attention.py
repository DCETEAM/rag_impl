import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        num_heads=12
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):

        output, weights = self.attention(
            x,
            x,
            x
        )

        return output
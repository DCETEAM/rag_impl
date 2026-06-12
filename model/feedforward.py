import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        hidden_dim=1536
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embed_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embed_dim
            )
        )

    def forward(
        self,
        x
    ):
        return self.network(x)
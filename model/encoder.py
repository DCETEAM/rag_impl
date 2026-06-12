import torch
import torch.nn as nn

from .attention import SelfAttention
from .feedforward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        num_heads=12,
        hidden_dim=1536
    ):
        super().__init__()

        self.attention = SelfAttention(
            embed_dim,
            num_heads
        )

        self.feedforward = FeedForward(
            embed_dim,
            hidden_dim
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):

        attn_output = self.attention(x)

        x = self.norm1(
            x + attn_output
        )

        ff_output = self.feedforward(x)

        x = self.norm2(
            x + ff_output
        )

        return x
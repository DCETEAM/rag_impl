import torch.nn as nn

class MLMHead(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        vocab_size=10000
    ):
        super().__init__()

        self.linear = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):
        return self.linear(x)
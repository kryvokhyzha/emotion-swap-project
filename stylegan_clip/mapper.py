import torch
import torch.nn as nn


class Mapper(nn.Module):

    def __init__(self, depth):
        super(Mapper, self).__init__()

        blocks = []
        for _ in range(14):
            block = nn.Sequential(
                nn.Linear(512, depth, bias=False),
                nn.BatchNorm1d(depth),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(depth, depth, bias=False),
                nn.BatchNorm1d(depth),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(depth, depth, bias=False),
                nn.BatchNorm1d(depth),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(depth, 512)
            )
            blocks.append(block)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.blocks = nn.ModuleList(blocks)
        self.apply(weights_init)

    def forward(self, w):
        """
        Arguments:
            w: a float tensor with shape [b, 14, 512].
        Returns:
            a float tensor with shape [b, 14, 512].
        """

        deltas = []
        for i, block in enumerate(self.blocks):
            deltas.append(block(w[:, i]))

        return torch.stack(deltas, dim=1).mul(0.1)

from torch import nn


class AdaIN(nn.Module):

    def __init__(self, channels, latent_size):
        super().__init__()
        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(latent_size, (channels + latent_size) // 2),
            nn.ELU(),
            nn.Linear((channels + latent_size) // 2, channels * 2)
        )

    def forward(self, x, dlatent):
        x = nn.InstanceNorm2d(self.channels)(x)
        style = self.linear(dlatent)
        style = style.view([-1, 2, x.size()[1]] + [1] * (len(x.size()) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]

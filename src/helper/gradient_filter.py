import numpy as np
import torch
from torch import nn


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class GradientMagnitudeFilter(nn.Module):

    def __init__(
            self,
            k_gaussian=3,
            mu=0,
            sigma=1,
            k_sobel=3,
            device='cpu',
    ):
        super(GradientMagnitudeFilter, self).__init__()

        self.device = device

        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=k_gaussian,
            padding=k_gaussian // 2,
            bias=False,
        )
        self.gaussian_filter.weight = torch.nn.Parameter(
            torch.from_numpy(gaussian_2D).unsqueeze(0).unsqueeze(0).float()
            .repeat(3, 3, 1, 1)
        )

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=k_sobel,
            padding=k_sobel // 2,
            bias=False,
        )
        self.sobel_filter_x.weight = torch.nn.Parameter(
            torch.from_numpy(sobel_2D).unsqueeze(0).unsqueeze(0).float()
            .repeat(3, 3, 1, 1)
        )

        self.sobel_filter_y = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=k_sobel,
            padding=k_sobel // 2,
            bias=False,
        )
        self.sobel_filter_y.weight = torch.nn.Parameter(
            torch.from_numpy(sobel_2D.T).unsqueeze(0).unsqueeze(0).float()
            .repeat(3, 3, 1, 1)
        )

    def forward(self, img):
        # compute gradients
        blurred = self.gaussian_filter(img)
        grad_x = self.sobel_filter_x(blurred).mean(dim=1, keepdim=True)
        grad_y = self.sobel_filter_y(blurred).mean(dim=1, keepdim=True)

        # compute magnitude
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5

        return grad_magnitude

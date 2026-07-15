"""
1D Convolutional Autoencoder, with configurable bottleneck width.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_channels=64):
        super().__init__()
        c = bottleneck_channels

        # Encoder: 360 -> 180 -> 90 -> 45, channels: 1 -> c/4 -> c/2 -> c
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c // 4),
            nn.ReLU(),

            nn.Conv1d(c // 4, c // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c // 2),
            nn.ReLU(),

            nn.Conv1d(c // 2, c, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(c, c // 2, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(c // 2),
            nn.ReLU(),

            nn.ConvTranspose1d(c // 2, c // 4, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(c // 4),
            nn.ReLU(),

            nn.ConvTranspose1d(c // 4, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
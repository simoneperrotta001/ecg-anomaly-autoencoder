"""
Dense (fully-connected) autoencoder, used as an architectural baseline
against the Conv1D model.
"""

import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=360, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        latent = self.encoder(x_flat)
        reconstruction = self.decoder(latent)
        return reconstruction.view(x.size(0), 1, -1)
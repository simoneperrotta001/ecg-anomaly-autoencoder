"""
Block 3: 1D Convolutional Autoencoder architecture.

Design: symmetric encoder-decoder. The encoder progressively compresses the
360-sample beat into a compact latent representation using strided
convolutions (downsampling). The decoder mirrors this with transposed
convolutions (upsampling) to reconstruct the original signal.
"""

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder: 360 -> 180 -> 90 -> 45, channels: 1 -> 16 -> 32 -> 64
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # 360 -> 180
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # 180 -> 90
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # 90 -> 45
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Decoder: mirrors the encoder, 45 -> 90 -> 180 -> 360, channels: 64 -> 32 -> 16 -> 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),  # 45 -> 90
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),  # 90 -> 180
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # 180 -> 360
            # No activation on the final layer: we want to reconstruct raw
            # (z-score normalized) signal values, which can be negative.
        )

    def forward(self, x):
        # x shape expected: (batch_size, 1, 360)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


if __name__ == "__main__":
    # Quick sanity check: verify input/output shapes match
    model = ConvAutoencoder()
    dummy_input = torch.randn(8, 1, 360)  # batch of 8 beats
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Shape mismatch! Check padding/output_padding."
    print("Shape check passed.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
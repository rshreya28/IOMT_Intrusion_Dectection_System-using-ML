"""
Deliberately small autoencoder: trains on NORMAL traffic only and flags an
alert when reconstruction error is high (i.e., traffic doesn't look like
anything it's seen before). This is more robust to *unseen* attack types than
a plain classifier, which can only recognize attacks it was explicitly trained
on -- useful for zero-day-style attacks in a fast-moving IoT threat landscape.

Kept small on purpose (a few thousand params) so it can be quantized and run
on constrained devices, not just a cloud server.
"""

import torch
import torch.nn as nn


class LightIDSAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

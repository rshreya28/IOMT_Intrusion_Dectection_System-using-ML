"""
model.py
Lightweight autoencoder for anomaly-based intrusion detection.

Trained ONLY on normal (benign) traffic. The idea: the model learns to
reconstruct normal traffic patterns well. When it sees something unusual
(an attack), it reconstructs it poorly -- that reconstruction error is
our anomaly signal.

Deliberately small (few thousand parameters) so it can realistically run
on constrained edge devices, matching the "small model" claim in the README.
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
        recon = self.decoder(z)
        return recon

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def reconstruction_error(self, x):
        """
        Returns per-sample reconstruction error (mean squared error per row).
        Used as the anomaly score: higher error = more likely to be an attack.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error
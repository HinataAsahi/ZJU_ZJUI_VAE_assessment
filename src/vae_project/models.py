from __future__ import annotations

from functools import reduce
from operator import mul

import torch
from torch import nn


class MLPVAE(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (1, 28, 28), hidden_dims: list[int] | None = None, latent_dim: int = 16):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [400, 200]
        self.input_shape = input_shape
        self.input_dim = reduce(mul, input_shape)
        self.latent_dim = latent_dim

        encoder_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.view(x.shape[0], -1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        return logits.view(z.shape[0], *self.input_shape)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return {"recon_logits": recon_logits, "mu": mu, "logvar": logvar, "z": z}

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_bce_with_logits(recon_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="none")
    per_sample = per_pixel.flatten(start_dim=1).sum(dim=1)
    return per_sample.mean()


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    per_dim = 1 + logvar - mu.pow(2) - logvar.exp()
    per_sample = -0.5 * per_dim.sum(dim=1)
    return per_sample.mean()


def vae_loss(
    recon_logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    reconstruction = reconstruction_bce_with_logits(recon_logits, target)
    kl = kl_divergence_standard_normal(mu, logvar)
    total = reconstruction + float(beta) * kl
    return {"total": total, "reconstruction": reconstruction, "kl": kl}

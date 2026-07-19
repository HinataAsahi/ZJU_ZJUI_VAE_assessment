from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from vae_project.utils import ensure_dir


@torch.no_grad()
def save_reconstruction_grid(
    model: torch.nn.Module, loader, device: torch.device, path: str | Path, max_images: int = 8
) -> None:
    model.eval()
    images, _labels = next(iter(loader))
    images = images[:max_images].to(device)
    output = model(images, sample=False)
    recon = torch.sigmoid(output["recon_logits"])
    comparison = torch.cat([images.cpu(), recon.cpu()], dim=0)
    output_path = Path(path)
    ensure_dir(output_path.parent)
    save_image(comparison, output_path, nrow=max_images)


@torch.no_grad()
def save_prior_samples(
    model: torch.nn.Module,
    device: torch.device,
    path: str | Path,
    sample_count: int = 64,
    latents: torch.Tensor | None = None,
) -> None:
    model.eval()
    z = torch.randn(sample_count, model.latent_dim, device=device) if latents is None else latents.to(device)
    samples = torch.sigmoid(model.decode(z)).cpu()
    output_path = Path(path)
    ensure_dir(output_path.parent)
    nrow = int(z.shape[0] ** 0.5)
    save_image(samples, output_path, nrow=max(nrow, 1))


def save_loss_curves(history: list[dict[str, float]], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_total"] for row in history], label="train total")
    plt.plot(epochs, [row["test_total"] for row in history], label="test total")
    plt.plot(epochs, [row["test_reconstruction"] for row in history], label="test reconstruction")
    plt.plot(epochs, [row["test_kl"] for row in history], label="test KL")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

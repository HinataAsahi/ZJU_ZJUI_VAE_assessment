from __future__ import annotations

from pathlib import Path

import torch

from vae_project.data import get_dataloaders
from vae_project.models import MLPVAE
from vae_project.train import evaluate_epoch
from vae_project.utils import ensure_dir, save_json, select_device
from vae_project.visualization import save_loss_curves, save_prior_samples, save_reconstruction_grid


def make_prior_latents(
    latent_dim: int, sample_count: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(sample_count, latent_dim, generator=generator).to(device)


def load_checkpoint(run_dir: str | Path, device: torch.device) -> tuple[MLPVAE, dict, list[dict[str, float]]]:
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = MLPVAE(input_shape=(1, 28, 28), hidden_dims=list(config["hidden_dims"]), latent_dim=int(config["latent_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    history = list(checkpoint.get("history", []))
    return model, config, history


def evaluate_run(run_dir: str | Path, device: str = "auto") -> dict:
    selected_device = select_device(device)
    model, config, history = load_checkpoint(run_dir, selected_device)
    _train_loader, test_loader = get_dataloaders(config)
    metrics = evaluate_epoch(model, test_loader, selected_device, beta=float(config["beta"]))
    figures_dir = ensure_dir(Path(run_dir) / "figures")
    save_reconstruction_grid(model, test_loader, selected_device, figures_dir / "reconstructions.png", max_images=8)
    sample_count = int(config.get("sample_count", 64))
    prior_latents = make_prior_latents(
        latent_dim=int(config["latent_dim"]),
        sample_count=sample_count,
        seed=int(config["seed"]),
        device=selected_device,
    )
    save_prior_samples(
        model,
        selected_device,
        figures_dir / "prior_samples.png",
        sample_count=sample_count,
        latents=prior_latents,
    )
    save_loss_curves(history, figures_dir / "loss_curves.png")
    summary = {
        "test_total": metrics["total"],
        "test_reconstruction": metrics["reconstruction"],
        "test_kl": metrics["kl"],
        "run_dir": str(run_dir),
        "device": str(selected_device),
    }
    save_json(summary, Path(run_dir) / "evaluation.json")
    return summary

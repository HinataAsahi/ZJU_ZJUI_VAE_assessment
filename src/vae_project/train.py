from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from vae_project.data import get_dataloaders
from vae_project.losses import vae_loss
from vae_project.models import MLPVAE
from vae_project.utils import ensure_dir, save_json, select_device, set_seed


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    count = 0
    for images, _labels in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        losses = vae_loss(output["recon_logits"], images, output["mu"], output["logvar"], beta=beta)
        losses["total"].backward()
        optimizer.step()
        batch_size = images.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module, loader, device: torch.device, beta: float) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    count = 0
    for images, _labels in loader:
        images = images.to(device)
        output = model(images, sample=True)
        losses = vae_loss(output["recon_logits"], images, output["mu"], output["logvar"], beta=beta)
        batch_size = images.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def fit(config: dict) -> dict:
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    run_dir = ensure_dir(config["output_dir"])
    train_loader, test_loader = get_dataloaders(config)
    model = MLPVAE(
        input_shape=(1, 28, 28),
        hidden_dims=list(config["hidden_dims"]),
        latent_dim=int(config["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        started = perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, beta=float(config["beta"]))
        test_metrics = evaluate_epoch(model, test_loader, device, beta=float(config["beta"]))
        elapsed = perf_counter() - started
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_kl": train_metrics["kl"],
            "test_total": test_metrics["total"],
            "test_reconstruction": test_metrics["reconstruction"],
            "test_kl": test_metrics["kl"],
            "epoch_seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_total={row['train_total']:.3f} "
            f"test_total={row['test_total']:.3f} "
            f"test_recon={row['test_reconstruction']:.3f} "
            f"test_kl={row['test_kl']:.3f}"
        )

    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        checkpoint_path,
    )
    save_json(config, Path(run_dir) / "config.json")
    save_json({"history": history}, Path(run_dir) / "metrics.json")
    return {"run_dir": str(run_dir), "history": history, "checkpoint": str(checkpoint_path)}

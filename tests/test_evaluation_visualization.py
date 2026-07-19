from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_project.config import validate_config
from vae_project.evaluate import evaluate_run
from vae_project.models import MLPVAE
from vae_project.train import evaluate_epoch, fit
from vae_project.visualization import save_reconstruction_grid


class RecordingMLPVAE(MLPVAE):
    def __init__(self):
        super().__init__(input_shape=(1, 28, 28), hidden_dims=[16], latent_dim=3)
        self.sample_requests: list[bool | None] = []

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        self.sample_requests.append(sample)
        return super().forward(x, sample=sample)


def make_loader() -> DataLoader:
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def test_evaluate_epoch_requests_sampled_latents():
    model = RecordingMLPVAE()

    evaluate_epoch(model, make_loader(), torch.device("cpu"), beta=1.0)

    assert model.sample_requests == [True, True]


def test_reconstruction_grid_requests_posterior_mean(tmp_path):
    model = RecordingMLPVAE()

    save_reconstruction_grid(model, make_loader(), torch.device("cpu"), tmp_path / "reconstructions.png")

    assert model.sample_requests == [False]


def test_evaluate_run_writes_figures(tmp_path):
    config = validate_config(
        {
            "run_name": "smoke",
            "dataset": "fake",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "outputs" / "smoke"),
            "batch_size": 8,
            "epochs": 1,
            "seed": 11,
            "device": "cpu",
            "beta": 1.0,
            "latent_dim": 4,
            "hidden_dims": [32],
            "train_limit": 16,
            "test_limit": 8,
            "download": False,
            "sample_count": 16,
        }
    )
    fit(config)

    summary = evaluate_run(config["output_dir"], device="cpu")
    run_dir = Path(config["output_dir"])

    assert summary["test_total"] > 0
    assert (run_dir / "figures" / "reconstructions.png").exists()
    assert (run_dir / "figures" / "prior_samples.png").exists()
    assert (run_dir / "figures" / "loss_curves.png").exists()
    assert (run_dir / "evaluation.json").exists()


def test_evaluate_run_reproduces_prior_samples_for_same_seed(tmp_path):
    config = validate_config(
        {
            "run_name": "deterministic",
            "dataset": "fake",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "outputs" / "deterministic"),
            "batch_size": 8,
            "epochs": 1,
            "seed": 23,
            "device": "cpu",
            "beta": 1.0,
            "latent_dim": 4,
            "hidden_dims": [32],
            "train_limit": 16,
            "test_limit": 8,
            "download": False,
            "sample_count": 16,
        }
    )
    fit(config)

    evaluate_run(config["output_dir"], device="cpu")
    prior_path = Path(config["output_dir"]) / "figures" / "prior_samples.png"
    first = prior_path.read_bytes()
    torch.manual_seed(999_999)
    evaluate_run(config["output_dir"], device="cpu")

    assert prior_path.read_bytes() == first

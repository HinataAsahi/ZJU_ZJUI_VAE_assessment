from pathlib import Path

from vae_project.config import validate_config
from vae_project.evaluate import evaluate_run
from vae_project.train import fit


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

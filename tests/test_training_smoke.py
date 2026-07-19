from pathlib import Path

from vae_project.config import validate_config
from vae_project.train import fit


def test_fit_writes_checkpoint_and_metrics(tmp_path):
    config = validate_config(
        {
            "run_name": "smoke",
            "dataset": "fake",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "outputs" / "smoke"),
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 1e-3,
            "seed": 7,
            "device": "cpu",
            "beta": 1.0,
            "latent_dim": 4,
            "hidden_dims": [32],
            "train_limit": 16,
            "test_limit": 8,
            "num_workers": 0,
            "download": False,
        }
    )

    result = fit(config)
    run_dir = Path(config["output_dir"])

    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert len(result["history"]) == 1
    assert result["history"][0]["train_total"] > 0
    assert result["history"][0]["test_total"] > 0

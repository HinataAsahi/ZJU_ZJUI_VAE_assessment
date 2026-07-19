import os
import subprocess
import sys
from pathlib import Path


def test_train_and_evaluate_cli_smoke(tmp_path):
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "outputs" / "cli_smoke"
    config_path.write_text(
        "\n".join(
            [
                "run_name: cli_smoke",
                "dataset: fake",
                f"data_dir: {tmp_path / 'data'}",
                f"output_dir: {output_dir}",
                "batch_size: 8",
                "epochs: 1",
                "learning_rate: 0.001",
                "seed: 5",
                "device: cpu",
                "beta: 1.0",
                "latent_dim: 4",
                "hidden_dims: [32]",
                "train_limit: 16",
                "test_limit: 8",
                "num_workers: 0",
                "download: false",
                "sample_count: 16",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(output_dir), "--device", "cpu"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "epoch=1" in train_result.stdout
    assert "test_total" in eval_result.stdout
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "evaluation.json").exists()

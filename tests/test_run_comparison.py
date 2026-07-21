import os
import subprocess
import sys
from pathlib import Path

from vae_project.comparison import build_markdown, load_run_summary
from vae_project.utils import save_json


def make_run(tmp_path: Path, name: str, beta: float, reconstruction: float, kl: float) -> Path:
    run_dir = tmp_path / name
    save_json(
        {
            "run_name": name,
            "dataset": "fashion_mnist",
            "beta": beta,
            "epochs": 20,
            "latent_dim": 16,
        },
        run_dir / "config.json",
    )
    save_json(
        {
            "device": "cuda",
            "run_dir": str(run_dir),
            "test_reconstruction": reconstruction,
            "test_kl": kl,
            "test_total": reconstruction + beta * kl,
        },
        run_dir / "evaluation.json",
    )
    save_json(
        {
            "history": [
                {
                    "epoch": 20,
                    "train_reconstruction": reconstruction - 2,
                    "train_kl": kl - 1,
                    "train_total": reconstruction + beta * kl - 3,
                    "test_reconstruction": reconstruction,
                    "test_kl": kl,
                    "test_total": reconstruction + beta * kl,
                }
            ]
        },
        run_dir / "metrics.json",
    )
    return run_dir


def test_load_run_summary_reads_config_metrics_and_evaluation(tmp_path):
    run_dir = make_run(tmp_path, "fashion_mnist_beta1", beta=1.0, reconstruction=228.38, kl=12.19)

    summary = load_run_summary(run_dir)

    assert summary.run_name == "fashion_mnist_beta1"
    assert summary.dataset == "fashion_mnist"
    assert summary.beta == 1.0
    assert summary.epochs == 20
    assert summary.final_epoch == 20
    assert summary.test_reconstruction == 228.38
    assert summary.test_kl == 12.19
    assert summary.device == "cuda"


def test_build_markdown_includes_run_table_and_figure_paths(tmp_path):
    beta1 = make_run(tmp_path, "fashion_mnist_beta1", beta=1.0, reconstruction=228.38, kl=12.19)
    beta0 = make_run(tmp_path, "fashion_mnist_beta0", beta=0.0, reconstruction=214.24, kl=297.92)
    summaries = [load_run_summary(beta1), load_run_summary(beta0)]

    markdown = build_markdown(summaries, title="Baseline comparison")

    assert "# Baseline comparison" in markdown
    assert "| run | dataset | beta | epochs | test reconstruction | test KL | test total |" in markdown
    assert "| fashion_mnist_beta1 | fashion_mnist | 1.0 | 20 | 228.38 | 12.19 | 240.57 |" in markdown
    assert "| fashion_mnist_beta0 | fashion_mnist | 0.0 | 20 | 214.24 | 297.92 | 214.24 |" in markdown
    assert "figures/reconstructions.png" in markdown
    assert "figures/prior_samples.png" in markdown


def test_compare_runs_cli_writes_markdown_file(tmp_path):
    beta1 = make_run(tmp_path, "fashion_mnist_beta1", beta=1.0, reconstruction=228.38, kl=12.19)
    beta0 = make_run(tmp_path, "fashion_mnist_beta0", beta=0.0, reconstruction=214.24, kl=297.92)
    output_path = tmp_path / "comparison.md"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_runs.py",
            str(beta1),
            str(beta0),
            "--title",
            "Baseline comparison",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "comparison.md" in result.stdout
    assert "| fashion_mnist_beta0 | fashion_mnist | 0.0 | 20 | 214.24 | 297.92 | 214.24 |" in output_path.read_text(
        encoding="utf-8"
    )

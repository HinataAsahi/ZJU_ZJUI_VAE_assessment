from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vae_project.utils import load_json


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    run_name: str
    dataset: str
    beta: float
    epochs: int
    latent_dim: int
    final_epoch: int
    device: str
    train_reconstruction: float | None
    train_kl: float | None
    train_total: float | None
    test_reconstruction: float | None
    test_kl: float | None
    test_total: float | None


def load_run_summary(run_dir: str | Path) -> RunSummary:
    path = Path(run_dir)
    config = load_json(path / "config.json")
    metrics = load_json(path / "metrics.json")
    evaluation = load_json(path / "evaluation.json") if (path / "evaluation.json").exists() else {}
    latest = _latest_history_row(metrics)

    return RunSummary(
        run_dir=path,
        run_name=str(config.get("run_name", path.name)),
        dataset=str(config.get("dataset", "unknown")),
        beta=float(config.get("beta", 0.0)),
        epochs=int(config.get("epochs", latest.get("epoch", 0))),
        latent_dim=int(config.get("latent_dim", 0)),
        final_epoch=int(latest.get("epoch", config.get("epochs", 0))),
        device=str(evaluation.get("device", config.get("device", "unknown"))),
        train_reconstruction=_optional_float(latest.get("train_reconstruction")),
        train_kl=_optional_float(latest.get("train_kl")),
        train_total=_optional_float(latest.get("train_total")),
        test_reconstruction=_optional_float(evaluation.get("test_reconstruction", latest.get("test_reconstruction"))),
        test_kl=_optional_float(evaluation.get("test_kl", latest.get("test_kl"))),
        test_total=_optional_float(evaluation.get("test_total", latest.get("test_total"))),
    )


def build_markdown(summaries: Iterable[RunSummary], title: str = "VAE Run Comparison") -> str:
    rows = list(summaries)
    lines = [
        f"# {title}",
        "",
        "## Metric Summary",
        "",
        "| run | dataset | beta | epochs | test reconstruction | test KL | test total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| {row.run_name}",
                    row.dataset,
                    _format_beta(row.beta),
                    str(row.final_epoch),
                    _format_metric(row.test_reconstruction),
                    _format_metric(row.test_kl),
                    f"{_format_metric(row.test_total)} |",
                ]
            )
        )

    lines.extend(["", "## Figure Files", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.run_name}",
                "",
                f"- Loss curves: `{row.run_dir / 'figures' / 'loss_curves.png'}`",
                f"- Reconstructions: `{row.run_dir / 'figures' / 'reconstructions.png'}`",
                f"- Prior samples: `{row.run_dir / 'figures' / 'prior_samples.png'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def compare_runs(run_dirs: Iterable[str | Path], title: str = "VAE Run Comparison") -> str:
    summaries = sorted(
        (load_run_summary(run_dir) for run_dir in run_dirs),
        key=lambda summary: (summary.beta, summary.run_name),
    )
    return build_markdown(summaries, title=title)


def _latest_history_row(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("metrics.json must contain a non-empty history list")
    latest = history[-1]
    if not isinstance(latest, dict):
        raise ValueError("metrics.json history entries must be objects")
    return latest


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_beta(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"

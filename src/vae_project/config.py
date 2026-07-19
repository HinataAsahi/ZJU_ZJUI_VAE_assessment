from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VALID_DATASETS = {"mnist", "fashion_mnist", "fake"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return validate_config(raw)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ["run_name", "dataset", "beta", "latent_dim"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    dataset = str(config["dataset"])
    if dataset not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"dataset must be one of: {valid}")

    run_name = str(config["run_name"])
    validated: dict[str, Any] = {
        "run_name": run_name,
        "dataset": dataset,
        "data_dir": str(config.get("data_dir", "data")),
        "output_dir": str(config.get("output_dir", f"outputs/{run_name}")),
        "batch_size": int(config.get("batch_size", 128)),
        "epochs": int(config.get("epochs", 5)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "seed": int(config.get("seed", 42)),
        "device": str(config.get("device", "auto")),
        "beta": float(config["beta"]),
        "latent_dim": int(config["latent_dim"]),
        "hidden_dims": [int(value) for value in config.get("hidden_dims", [400, 200])],
        "train_limit": _optional_int(config.get("train_limit")),
        "test_limit": _optional_int(config.get("test_limit")),
        "num_workers": int(config.get("num_workers", 0)),
        "download": bool(config.get("download", True)),
        "sample_count": int(config.get("sample_count", 64)),
    }

    if validated["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if validated["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if validated["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if validated["latent_dim"] <= 0:
        raise ValueError("latent_dim must be positive")
    if not validated["hidden_dims"]:
        raise ValueError("hidden_dims must contain at least one layer size")
    if any(value <= 0 for value in validated["hidden_dims"]):
        raise ValueError("hidden_dims must contain positive layer sizes")

    return validated


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

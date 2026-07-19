from __future__ import annotations

import argparse

from vae_project.config import load_config, validate_config
from vae_project.train import fit
from vae_project.utils import configure_matplotlib_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small VAE.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib_cache()
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
        config = validate_config(config)
    result = fit(config)
    print(f"run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from vae_project.evaluate import evaluate_run
from vae_project.utils import configure_matplotlib_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE run.")
    parser.add_argument("--run-dir", required=True, help="Directory containing checkpoint.pt and config.json.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib_cache()
    args = parse_args()
    summary = evaluate_run(args.run_dir, device=args.device)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

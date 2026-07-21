from __future__ import annotations

import argparse
from pathlib import Path

from vae_project.comparison import compare_runs
from vae_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VAE experiment run directories.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing config.json, metrics.json, and evaluation.json.")
    parser.add_argument("--title", default="VAE Run Comparison", help="Markdown title.")
    parser.add_argument("--output", help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = compare_runs(args.run_dirs, title=args.title)
    if args.output:
        output_path = Path(args.output)
        ensure_dir(output_path.parent)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {output_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()

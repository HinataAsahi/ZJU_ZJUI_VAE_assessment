# Repository Guidelines

## Project Structure & Module Organization

The core Python package lives in `src/vae_project/`. Key modules are `models.py` for the MLP VAE, `losses.py` for reconstruction and KL objectives, `data.py` for datasets and loaders, `train.py` and `evaluate.py` for orchestration, `visualization.py` for figures, and `config.py`/`utils.py` for configuration, seeding, device selection, and I/O helpers.

CLI entry points are in `scripts/`: use `scripts/train.py` for training and `scripts/evaluate.py` for post-training evaluation. Experiment settings live in `configs/*.yaml`. Tests are in `tests/test_*.py`. The learning notebook is `notebooks/01_vae_basics.ipynb`. Local outputs, datasets, checkpoints, caches, and internal planning files are intentionally ignored by Git.

## Build, Test, and Development Commands

Set up dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full test suite:

```bash
PYTHONPATH=src pytest -v
```

Run the offline smoke workflow, which uses `FakeData` and does not download datasets:

```bash
PYTHONPATH=src python scripts/train.py --config configs/mnist_smoke.yaml
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/mnist_smoke --device cpu
```

For Fashion-MNIST experiments, use `--device auto` or `--device cuda` on a CUDA-enabled machine.

## Coding Style & Naming Conventions

Use Python 3.11 style with 4-space indentation. Prefer `snake_case` for functions, variables, and config keys; use `PascalCase` for classes such as model types. Keep modules focused and follow the existing small-function style. Use typed dataclasses or explicit dictionaries where the current code already does. Keep comments sparse and reserve them for non-obvious logic.

## Testing Guidelines

This project uses `pytest`. Name test files `test_*.py` and test functions `test_*`. New behavior should include a focused unit test; changes to CLI, data loading, training, evaluation, or notebook execution should include an integration or smoke test. Tests should stay CPU-friendly, deterministic, and offline by default. Use `FakeData` for test workflows that must not depend on network downloads.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `docs:`, `fix:`, `test:`, `feat:`, or `chore:` followed by a concise imperative summary. Keep each commit focused.

Pull requests should describe the change, list verification commands run, and include relevant generated figures or metric summaries for experiment-facing updates. Do not commit `outputs/`, `data/`, `*.pt`, `*.pth`, course PDFs, caches, or internal spec/plan files.

## Configuration & Reproducibility

Prefer YAML configs under `configs/` over hard-coded experiment parameters. Keep seeds explicit when adding experiments. Avoid machine-specific paths; generated artifacts should stay under `outputs/` and remain untracked.

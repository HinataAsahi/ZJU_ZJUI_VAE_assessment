# ZJU ZJUI VAE Assessment

This repository contains a learning-first PyTorch implementation of a small Variational Autoencoder (VAE) for the PIL Lab 2026 summer assessment.

## What This Project Builds

The code implements a small MLP VAE for MNIST-style 28x28 grayscale images. It supports:

- training and test evaluation,
- image reconstruction,
- prior sampling from `N(0, I)`,
- loss logging for total loss, reconstruction loss, and KL loss,
- the required comparison between `beta=1` and `beta=0`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch is already installed, keep the existing PyTorch build and install any missing packages from `requirements.txt`.

## First Learning Step

Open:

```text
notebooks/01_vae_basics.ipynb
```

Read it in order. The notebook introduces:

1. why VAE is different from a plain autoencoder,
2. why the encoder outputs `mu` and `logvar`,
3. how `z = mu + std * eps` keeps sampling differentiable,
4. why the loss is reconstruction plus `beta * KL`,
5. why `beta=0` can reconstruct well but sample poorly.

Launch Jupyter from the repository root or the `notebooks/` directory. The first
code cell locates the repository and adds `src/` to Python's import path, so the
notebook does not require a separate `PYTHONPATH` setting.

## Local Smoke Test

The smoke config uses `FakeData`, so it does not download MNIST or Fashion-MNIST:

```bash
PYTHONPATH=src python scripts/train.py --config configs/mnist_smoke.yaml
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/mnist_smoke --device cpu
```

Expected generated files:

```text
outputs/mnist_smoke/checkpoint.pt
outputs/mnist_smoke/config.json
outputs/mnist_smoke/metrics.json
outputs/mnist_smoke/evaluation.json
outputs/mnist_smoke/figures/reconstructions.png
outputs/mnist_smoke/figures/prior_samples.png
outputs/mnist_smoke/figures/loss_curves.png
```

## Required Fashion-MNIST Experiments

Run these on a GPU machine when possible.

Standard VAE:

```bash
PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta1.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta1 --device auto
```

KL ablation:

```bash
PYTHONPATH=src python scripts/train.py --config configs/fashion_mnist_beta0.yaml --device auto
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/fashion_mnist_beta0 --device auto
```

## Tests

```bash
PYTHONPATH=src pytest -v
```

## Notes For The Report

The report should compare `beta=1` and `beta=0` using:

- test reconstruction loss,
- test KL loss,
- reconstruction grids,
- prior sampling grids,
- loss curves.

Do not compare `test_total` directly across beta values: each objective weights the KL term differently. Compare reconstruction loss, KL loss, generated samples, and qualitative grids instead.

The analysis should explain the tradeoff: removing KL usually helps reconstruction but damages the match between encoded latent vectors and the standard normal prior, so random prior samples become less reliable.

## Git Hygiene

The repository ignores local data, outputs, checkpoints, and internal planning notes. Generated files under `outputs/` remain local and must not be committed to public Git history.

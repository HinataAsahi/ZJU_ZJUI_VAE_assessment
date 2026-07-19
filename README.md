# ZJU ZJUI VAE Assessment

This repository contains a learning-first PyTorch implementation of a small Variational Autoencoder (VAE) for the PIL Lab 2026 summer assessment.

The first goal is to understand the VAE pipeline:

1. encode an image into `mu` and `logvar`,
2. sample a latent vector with the reparameterization trick,
3. decode the latent vector into an image,
4. optimize reconstruction loss plus `beta * KL(q(z|x) || p(z))`,
5. compare `beta=1` with `beta=0`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an existing PyTorch environment, install only the missing packages from `requirements.txt`.

## Learning Path

Start with:

```text
notebooks/01_vae_basics.ipynb
```

Then use the CLI scripts for reproducible runs.

## Quick Smoke Test

```bash
PYTHONPATH=src python scripts/train.py --config configs/mnist_smoke.yaml
PYTHONPATH=src python scripts/evaluate.py --run-dir outputs/mnist_smoke
```

## Required Experiments

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

Generated outputs are written under `outputs/` and are not tracked by Git.

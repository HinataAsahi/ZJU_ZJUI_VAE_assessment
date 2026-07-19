from pathlib import Path

import torch

from vae_project.config import load_config, validate_config
from vae_project.utils import ensure_dir, load_json, save_json, select_device, set_seed


def test_validate_config_fills_defaults():
    config = validate_config(
        {
            "run_name": "smoke",
            "dataset": "fake",
            "beta": 1.0,
            "latent_dim": 4,
        }
    )

    assert config["run_name"] == "smoke"
    assert config["dataset"] == "fake"
    assert config["beta"] == 1.0
    assert config["latent_dim"] == 4
    assert config["batch_size"] == 128
    assert config["epochs"] == 5
    assert config["hidden_dims"] == [400, 200]
    assert config["output_dir"] == "outputs/smoke"


def test_validate_config_rejects_unknown_dataset():
    try:
        validate_config({"run_name": "bad", "dataset": "cifar10", "beta": 1.0, "latent_dim": 4})
    except ValueError as exc:
        assert "dataset must be one of" in str(exc)
    else:
        raise AssertionError("validate_config accepted an unknown dataset")


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("run_name: smoke\ndataset: fake\nbeta: 0\nlatent_dim: 2\n", encoding="utf-8")

    config = load_config(path)

    assert config["run_name"] == "smoke"
    assert config["dataset"] == "fake"
    assert config["beta"] == 0.0
    assert config["latent_dim"] == 2


def test_seed_makes_torch_random_repeatable():
    set_seed(123)
    first = torch.rand(3)
    set_seed(123)
    second = torch.rand(3)

    assert torch.equal(first, second)


def test_select_device_cpu_and_auto():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "cuda", "mps"}


def test_json_helpers_and_ensure_dir(tmp_path):
    output_dir = ensure_dir(tmp_path / "nested")
    path = output_dir / "metrics.json"

    save_json({"loss": 1.25}, path)

    assert path.exists()
    assert load_json(path) == {"loss": 1.25}

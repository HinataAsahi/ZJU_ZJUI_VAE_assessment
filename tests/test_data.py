import torch

from vae_project.data import build_dataset, get_dataloaders


def test_fake_dataset_limit_and_shape(tmp_path):
    dataset = build_dataset("fake", root=str(tmp_path), train=True, download=False, limit=11)

    image, label = dataset[0]

    assert len(dataset) == 11
    assert image.shape == (1, 28, 28)
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0
    assert isinstance(label, int)


def test_get_dataloaders_returns_batches(tmp_path):
    config = {
        "dataset": "fake",
        "data_dir": str(tmp_path),
        "batch_size": 4,
        "train_limit": 8,
        "test_limit": 6,
        "num_workers": 0,
        "download": False,
    }

    train_loader, test_loader = get_dataloaders(config)
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (4, 1, 28, 28)
    assert train_labels.shape == (4,)
    assert test_images.shape == (4, 1, 28, 28)
    assert test_labels.shape == (4,)
    assert train_images.dtype == torch.float32

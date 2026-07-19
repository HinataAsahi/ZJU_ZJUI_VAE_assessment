from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_dataset(name: str, root: str, train: bool, download: bool = True, limit: int | None = None) -> Dataset:
    transform = transforms.ToTensor()
    normalized_name = name.lower()
    if normalized_name == "mnist":
        dataset: Dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)
    elif normalized_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    elif normalized_name == "fake":
        size = 128 if train else 64
        random_offset = 0 if train else 10_000
        dataset = datasets.FakeData(
            size=size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=random_offset,
        )
    else:
        raise ValueError("dataset must be one of: mnist, fashion_mnist, fake")

    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
    return dataset


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset(
        config["dataset"],
        root=config.get("data_dir", "data"),
        train=True,
        download=bool(config.get("download", True)),
        limit=config.get("train_limit"),
    )
    test_dataset = build_dataset(
        config["dataset"],
        root=config.get("data_dir", "data"),
        train=False,
        download=bool(config.get("download", True)),
        limit=config.get("test_limit"),
    )
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

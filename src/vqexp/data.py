from __future__ import annotations

import os
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets, transforms


def _subset(dataset, size: int, seed: int):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


class _HFDatasetAdapter(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        columns = set(hf_dataset.column_names)
        if "image" in columns:
            self.image_key = "image"
        elif "img" in columns:
            self.image_key = "img"
        else:
            raise ValueError(
                f"Expected an image column named 'image' or 'img', got: {hf_dataset.column_names}"
            )

        if "label" in columns:
            self.label_key = "label"
        else:
            raise ValueError(
                f"Expected a label column named 'label', got: {hf_dataset.column_names}"
            )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        example = self.hf_dataset[index]
        image = example[self.image_key]
        label = int(example[self.label_key])

        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    os.makedirs(cfg["data_root"], exist_ok=True)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if cfg["dataset"] == "cifar10":

        train_hf = load_dataset(
            "uoft-cs/cifar10",
            split="train",
            cache_dir=cfg["data_root"],
        )
        test_hf = load_dataset(
            "uoft-cs/cifar10",
            split="test",
            cache_dir=cfg["data_root"],
        )

        train_set = _HFDatasetAdapter(train_hf, transform=transform_train)
        test_set = _HFDatasetAdapter(test_hf, transform=transform_test)

    elif cfg["dataset"] == "fake":
        train_size = cfg["train_subset"] if cfg["train_subset"] > 0 else 512
        test_size = cfg["test_subset"] if cfg["test_subset"] > 0 else 256
        train_set = tv_datasets.FakeData(
            size=train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform_train,
            random_offset=0,
        )
        test_set = tv_datasets.FakeData(
            size=test_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform_test,
            random_offset=train_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}")

    train_set = _subset(train_set, int(cfg["train_subset"]), int(cfg["seed"]))
    test_set = _subset(test_set, int(cfg["test_subset"]), int(cfg["seed"]) + 1)

    pin_memory = torch.cuda.is_available()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg["seed"]))

    common = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": pin_memory,
        "persistent_workers": int(cfg["num_workers"]) > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader
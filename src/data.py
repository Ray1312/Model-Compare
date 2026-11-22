"""Dataset split used by all eight notebooks.

Fashion-MNIST train and test originals are concatenated before a single
seeded split into 56k train / 7k validation / 7k test.
"""

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128


def create_datasets(root="./data", seed=SEED, download=True):
    transform = transforms.ToTensor()
    original_train = datasets.FashionMNIST(
        root=root, train=True, download=download, transform=transform
    )
    original_test = datasets.FashionMNIST(
        root=root, train=False, download=download, transform=transform
    )
    full_dataset = ConcatDataset([original_train, original_test])
    return random_split(
        full_dataset,
        [56000, 7000, 7000],
        generator=torch.Generator().manual_seed(seed),
    )


def create_loaders(root="./data", batch_size=BATCH_SIZE, seed=SEED, download=True):
    train_dataset, val_dataset, test_dataset = create_datasets(
        root=root, seed=seed, download=download
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )

"""
Data pipelines
==============
Every dataset yields three loaders:

    train   – θ (network weights) is trained on this
    heldout – disjoint split of the official TRAIN set, used ONLY by the
              outer loop (φ) and for monitoring; never for model selection
    test    – official test / val set; reported once, at the end of training

The held-out split is fixed (split_seed=0) so every method and every seed
sees exactly the same partition. Non-bilevel baselines simply never touch it.

CIFAR images stay 32×32 in the loader and are upsampled on the GPU by the
model's InputResize, which is far cheaper than resizing in workers.
Pretrained ImageNet backbones → ImageNet normalization everywhere.
"""

import copy
import os

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "smoketest": 10,
}


def _split_indices(
    targets, per_class: int = None, fraction: float = None, seed: int = 0
):
    g = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(targets)
    if per_class is not None:
        held = []
        for c in targets.unique():
            idx = (targets == c).nonzero().flatten()
            held.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
        held = torch.cat(held)
    else:
        perm = torch.randperm(len(targets), generator=g)
        held = perm[: int(len(targets) * fraction)]
    mask = torch.ones(len(targets), dtype=torch.bool)
    mask[held] = False
    return mask.nonzero().flatten().tolist(), held.sort().values.tolist()


def _cifar(name, root, download):
    from torchvision import datasets, transforms

    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    t_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    t_eval = transforms.Compose([transforms.ToTensor(), norm])
    train_full = cls(root, train=True, download=download, transform=t_train)
    held_full = cls(root, train=True, download=False, transform=t_eval)
    test = cls(root, train=False, download=download, transform=t_eval)
    tr_idx, held_idx = _split_indices(train_full.targets, fraction=0.1)
    return Subset(train_full, tr_idx), Subset(held_full, held_idx), test


def _imagenet(root):
    """Expects root/train/<wnid>/*.JPEG and root/val/<wnid>/*.JPEG."""
    from torchvision import datasets, transforms

    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    t_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    t_eval = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm,
        ]
    )
    train_full = datasets.ImageFolder(
        os.path.join(root, "train"), transform=t_train
    )
    held_full = copy.copy(train_full)  # shares the file index, no rescan
    held_full.transform = t_eval
    test = datasets.ImageFolder(os.path.join(root, "val"), transform=t_eval)
    tr_idx, held_idx = _split_indices(train_full.targets, per_class=10)
    return Subset(train_full, tr_idx), Subset(held_full, held_idx), test


def _smoketest(n: int = 256):
    """Random tensors for the CPU smoke tests only; never used for results."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    ds = TensorDataset(x, y)
    return ds, ds, ds


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a NumPy/random seed derived from torch's."""
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    workers: int = 8,
    download: bool = False,
    seed: int = 0,
):
    """Return (train, heldout, test) DataLoaders.

    Shuffling and augmentation randomness are seeded from ``seed`` so a run
    is repeatable; the train/held-out split itself is fixed (split_seed=0)
    and identical for every seed and method.
    """
    if dataset in ("cifar10", "cifar100"):
        train, held, test = _cifar(dataset, root, download)
    elif dataset == "imagenet":
        train, held, test = _imagenet(root)
    elif dataset == "smoketest":
        train, held, test = _smoketest()
        workers = 0
    else:
        raise ValueError(dataset)

    kw = dict(
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
    )

    def gen(offset: int) -> torch.Generator:
        return torch.Generator().manual_seed(seed * 1000 + offset)

    return (
        DataLoader(
            train,
            batch_size,
            shuffle=True,
            drop_last=True,
            generator=gen(1),
            **kw,
        ),
        DataLoader(
            held,
            batch_size,
            shuffle=True,
            drop_last=True,
            generator=gen(2),
            **kw,
        ),
        DataLoader(test, batch_size * 2, shuffle=False, **kw),
    )

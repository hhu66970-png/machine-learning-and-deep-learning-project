"""Downstream datasets for the PEFT study.

All datasets are loaded via torchvision and resized to 224x224 to match the
ImageNet-pretrained ViT. Train uses light augmentation; test uses center crop.
Supports train-set sub-sampling (train_fraction) for data-efficiency experiments.
"""
import os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision.datasets as D
import torchvision.transforms as T

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASET_INFO = {
    "cifar100": 100,
    "flowers102": 102,
    "pets": 37,
    "dtd": 47,
    "cifar10": 10,
    "svhn": 10,
    "eurosat": 10,
    "fgvc_aircraft": 100,
    "gtsrb": 43,
}

# friendly names for figures/tables
DATASET_DISPLAY = {
    "cifar100": "CIFAR-100", "flowers102": "Flowers-102", "pets": "Pets", "dtd": "DTD",
    "cifar10": "CIFAR-10", "svhn": "SVHN", "eurosat": "EuroSAT",
    "fgvc_aircraft": "Aircraft", "gtsrb": "GTSRB",
}


def build_transforms(mean, std, img_size=224):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf


def _split_indices(n, frac_train=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = int(n * frac_train)
    return perm[:k], perm[k:]


def _build(name, train_tf, test_tf, root):
    if name == "cifar100":
        tr = D.CIFAR100(root, train=True, transform=train_tf, download=False)
        te = D.CIFAR100(root, train=False, transform=test_tf, download=False)
    elif name == "cifar10":
        tr = D.CIFAR10(root, train=True, transform=train_tf, download=False)
        te = D.CIFAR10(root, train=False, transform=test_tf, download=False)
    elif name == "svhn":
        tr = D.SVHN(root, split="train", transform=train_tf, download=False)
        te = D.SVHN(root, split="test", transform=test_tf, download=False)
    elif name == "gtsrb":
        tr = D.GTSRB(root, split="train", transform=train_tf, download=False)
        te = D.GTSRB(root, split="test", transform=test_tf, download=False)
    elif name == "fgvc_aircraft":
        tr = D.FGVCAircraft(root, split="trainval", annotation_level="variant",
                            transform=train_tf, download=False)
        te = D.FGVCAircraft(root, split="test", annotation_level="variant",
                            transform=test_tf, download=False)
    elif name == "eurosat":
        # EuroSAT has no official split -> deterministic 80/20 split with disjoint indices
        full_tr = D.EuroSAT(root, transform=train_tf, download=False)
        full_te = D.EuroSAT(root, transform=test_tf, download=False)
        idx_tr, idx_te = _split_indices(len(full_tr), 0.8, seed=42)
        tr, te = Subset(full_tr, idx_tr), Subset(full_te, idx_te)
    elif name == "flowers102":
        tr = ConcatDataset([
            D.Flowers102(root, split="train", transform=train_tf, download=False),
            D.Flowers102(root, split="val", transform=train_tf, download=False),
        ])
        te = D.Flowers102(root, split="test", transform=test_tf, download=False)
    elif name == "pets":
        tr = D.OxfordIIITPet(root, split="trainval", transform=train_tf, download=False)
        te = D.OxfordIIITPet(root, split="test", transform=test_tf, download=False)
    elif name == "dtd":
        tr = ConcatDataset([
            D.DTD(root, split="train", transform=train_tf, download=False),
            D.DTD(root, split="val", transform=train_tf, download=False),
        ])
        te = D.DTD(root, split="test", transform=test_tf, download=False)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return tr, te


def _subsample(dataset, fraction, seed=42):
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = max(1, int(n * fraction))
    return Subset(dataset, perm[:k])


def get_loaders(name, mean, std, batch_size=64, num_workers=8, img_size=224,
                root=DATA_ROOT, train_fraction=1.0, seed=42):
    train_tf, test_tf = build_transforms(mean, std, img_size)
    train_ds, test_ds = _build(name, train_tf, test_tf, root)
    if train_fraction < 1.0:
        train_ds = _subsample(train_ds, train_fraction, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(batch_size, 256), shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, DATASET_INFO[name]

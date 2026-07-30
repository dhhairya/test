"""
CropGuard AI — PlantVillage Dataset Loader

Expects data in PlantVillage format:
    data/plantvillage/
        Apple___Apple_scab/
            image001.jpg
            image002.jpg
        Apple___healthy/
            ...
        Tomato___Late_blight/
            ...

Download from:
    Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    TF Datasets: tfds.load('plant_village')

Usage:
    from training.dataset import get_dataloaders
    train_loader, val_loader, classes = get_dataloaders('data/plantvillage', batch_size=32)
"""
import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ── ImageNet normalization (used by MobileNetV2 pretrained backbone) ──────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Augmentation pipeline (training) ─────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1),  # Simulates occlusion / dirty lens
])

# ── Deterministic pipeline (validation / inference) ──────────────────────────
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train and validation DataLoaders from a PlantVillage-structured directory.

    Args:
        data_dir:    Path to directory containing one subdirectory per class.
        batch_size:  Images per batch.
        val_split:   Fraction of data to reserve for validation (0.0–0.5).
        num_workers: DataLoader worker processes.
        seed:        Random seed for reproducible split.

    Returns:
        (train_loader, val_loader, class_names)

    Raises:
        FileNotFoundError if data_dir doesn't exist.
        ValueError if data_dir contains no class subdirectories.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Download PlantVillage from:\n"
            "  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
            "and extract to data/plantvillage/"
        )

    # Load full dataset with training transforms (we'll split it)
    full_dataset = datasets.ImageFolder(str(data_dir), transform=TRAIN_TRANSFORMS)

    if len(full_dataset.classes) == 0:
        raise ValueError(f"No classes found in {data_dir}. Check directory structure.")

    class_names = full_dataset.classes
    n_classes   = len(class_names)
    n_total     = len(full_dataset)
    n_val       = int(n_total * val_split)
    n_train     = n_total - n_val

    logger.info(f"Dataset: {n_classes} classes, {n_total} images "
                f"({n_train} train / {n_val} val)")

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Override val transforms without altering the split indices
    val_ds.dataset = datasets.ImageFolder(str(data_dir), transform=VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, class_names


def get_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Pass to nn.CrossEntropyLoss(weight=...) during training.
    """
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1

    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * len(dataset.classes)
    return weights.to(device)


def save_class_mapping(class_names: List[str], output_path: str):
    """Save class index → name mapping to JSON (loaded at inference time)."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved {len(class_names)} class names to {output_path}")

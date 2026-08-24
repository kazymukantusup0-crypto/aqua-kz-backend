"""PyTorch datasets for seven-channel drought segmentation patches."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from app.config import DATASET_ROOT, PATCH_CHANNELS


def training_augmentations() -> A.Compose:
    """Spatial augmentations that preserve pixel-wise drought masks."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )


def normalize_patch(image: np.ndarray) -> np.ndarray:
    """Map reflectance and NDVI ranges into stable 0..1 model inputs."""
    if image.shape[0] != len(PATCH_CHANNELS):
        raise ValueError(f"Expected {len(PATCH_CHANNELS)} channels, received {image.shape[0]}.")
    normalized = image.astype(np.float32, copy=True)
    normalized[:6] = np.clip((normalized[:6] + 0.2) / 1.2, 0.0, 1.0)
    normalized[6] = np.clip((normalized[6] + 1.0) / 2.0, 0.0, 1.0)
    return normalized


class NpyDroughtPatchDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load paired `C,H,W` images and `H,W` binary labels from `.npy` files."""

    def __init__(self, images_dir: Path, labels_dir: Path, transform: A.Compose | None = None) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_paths = sorted(images_dir.glob("*.npy"))
        missing_labels = [path.stem for path in self.image_paths if not (labels_dir / path.name).is_file()]
        if missing_labels:
            raise ValueError(f"Missing label patches for: {', '.join(missing_labels[:5])}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]
        image = np.load(image_path).astype(np.float32)
        label = np.load(self.labels_dir / image_path.name).astype(np.float32)
        if image.ndim != 3:
            raise ValueError(f"Image patch {image_path.name} must have C,H,W dimensions.")
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if label.ndim != 2 or image.shape[1:] != label.shape:
            raise ValueError(f"Image and label shapes do not match for {image_path.name}.")

        image = normalize_patch(image)
        if self.transform:
            transformed = self.transform(image=np.moveaxis(image, 0, -1), mask=label)
            image = np.moveaxis(np.asarray(transformed["image"], dtype=np.float32), -1, 0)
            label = np.asarray(transformed["mask"], dtype=np.float32)

        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(label[None, ...]))


def create_dataloaders(
    dataset_root: Path = DATASET_ROOT,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create train/validation/test loaders from the documented patch folder layout."""
    def dataset_for(split: str, transform: A.Compose | None) -> NpyDroughtPatchDataset:
        return NpyDroughtPatchDataset(
            dataset_root / split / "images",
            dataset_root / split / "labels",
            transform=transform,
        )

    train = dataset_for("train", training_augmentations())
    val = dataset_for("val", None)
    test = dataset_for("test", None)
    if not len(train) or not len(val):
        raise ValueError("Training and validation splits must contain at least one paired .npy patch.")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    )

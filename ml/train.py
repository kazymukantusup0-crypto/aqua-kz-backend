"""Train a seven-channel U-Net for educational binary drought segmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from app.config import DATASET_ROOT, PATCH_CHANNELS, SEGMENTATION_MODEL_PATH
from ml.dataset import create_dataloaders


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_workers: int = 0
    encoder_name: str = "resnet34"


def build_unet(encoder_name: str = "resnet34") -> nn.Module:
    """Create U-Net without downloading pretrained weights, configured for seven bands."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=len(PATCH_CHANNELS),
        classes=1,
        activation=None,
    )


class BCEDiceLoss(nn.Module):
    """Balanced BCE-with-logits plus soft Dice loss for sparse binary masks."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return bce + dice_loss


def _run_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    batches = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = criterion(model(images), masks)
            if training:
                loss.backward()
                optimizer.step()
        loss_total += float(loss.detach().cpu())
        batches += 1
    if not batches:
        raise ValueError("DataLoader contains no batches.")
    return loss_total / batches


def _evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    """Calculate loss and thresholded binary-segmentation metrics for one split."""
    model.eval()
    loss_total = 0.0
    batches = 0
    true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss_total += float(criterion(logits, masks).detach().cpu())
            predicted = torch.sigmoid(logits) >= 0.5
            target = masks >= 0.5
            true_positive += int((predicted & target).sum().item())
            false_positive += int((predicted & ~target).sum().item())
            false_negative += int((~predicted & target).sum().item())
            batches += 1
    if not batches:
        raise ValueError("DataLoader contains no batches.")
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "loss": loss_total / batches,
        "precision": precision,
        "recall": recall,
        "dice": (2 * true_positive) / max(2 * true_positive + false_positive + false_negative, 1),
        "iou": true_positive / max(true_positive + false_positive + false_negative, 1),
    }


def train_segmentation(
    config: TrainingConfig = TrainingConfig(),
    dataset_root: Path = DATASET_ROOT,
    checkpoint_path: Path = SEGMENTATION_MODEL_PATH,
    progress_callback: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> dict[str, float | int | str]:
    """Train U-Net and persist the checkpoint with the lowest validation loss."""
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(config.encoder_name).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_validation_loss = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        training_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = _run_epoch(model, val_loader, criterion, device)
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder_name": config.encoder_name,
                    "in_channels": len(PATCH_CHANNELS),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
        if progress_callback:
            progress_callback(
                {
                    "epoch": epoch,
                    "train_loss": training_loss,
                    "validation_loss": validation_loss,
                    "best_checkpoint": improved,
                }
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = _evaluate(model, test_loader, criterion, device)
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_validation_loss": round(best_validation_loss, 6),
        "epochs": config.epochs,
        "device": str(device),
        "test_loss": round(test_metrics["loss"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
        "test_dice": round(test_metrics["dice"], 6),
        "test_iou": round(test_metrics["iou"], 6),
    }

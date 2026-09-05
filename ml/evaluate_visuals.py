"""Create reproducible visual and per-patch evaluation artifacts for the completed U-Net run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from app.config import DATASET_ROOT, SEGMENTATION_MODEL_PATH
from ml.dataset import NpyDroughtPatchDataset
from ml.train import build_unet


PREDICTION_THRESHOLD = 0.5
PREDICTIONS_DIR = Path(__file__).resolve().parent / "weights" / "predictions"

# The completed run used these nine test files. They are retained explicitly because
# a later export appended more patches to data/patches/test after that run completed.
COMPLETED_RUN_TEST_PATCH_IDS = (
    "akmola_2024_00006",
    "akmola_2024_00024",
    "akmola_2024_00033",
    "akmola_2024_00038",
    "akmola_2024_00053",
    "karaganda_2024_00000",
    "karaganda_2024_00011",
    "karaganda_2024_00012",
    "karaganda_2024_00016",
)


@dataclass(frozen=True)
class PatchMetrics:
    patch_id: str
    dice: float
    iou: float
    precision: float
    recall: float
    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int


def _metrics(patch_id: str, predicted: np.ndarray, target: np.ndarray) -> PatchMetrics:
    predicted = predicted.astype(bool)
    target = target.astype(bool)
    true_positive = int(np.logical_and(predicted, target).sum())
    false_positive = int(np.logical_and(predicted, ~target).sum())
    false_negative = int(np.logical_and(~predicted, target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    dice = (2 * true_positive) / max(2 * true_positive + false_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return PatchMetrics(
        patch_id=patch_id,
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        threshold=PREDICTION_THRESHOLD,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _false_color(image: np.ndarray) -> np.ndarray:
    """Render normalized NIR/Red/Green bands as R/G/B respectively."""
    return np.clip(np.stack((image[3], image[2], image[1]), axis=-1), 0.0, 1.0)


def _save_panels(
    output_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    patch_id: str,
) -> None:
    visual_input = _false_color(image)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    figure.suptitle(patch_id, fontsize=14)

    axes[0].imshow(visual_input)
    axes[0].set_title("False color (NIR / Red / Green)")

    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth drought mask")

    probability_image = axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Predicted drought probability")
    figure.colorbar(probability_image, ax=axes[2], fraction=0.046, pad=0.04, label="Probability")

    axes[3].imshow(visual_input)
    axes[3].imshow(np.ma.masked_where(~prediction, prediction), cmap="Reds", vmin=0, vmax=1, alpha=0.58)
    axes[3].set_title(f"Prediction overlay (≥ {PREDICTION_THRESHOLD:.1f})")

    for axis in axes:
        axis.set_axis_off()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_csv(metrics: list[PatchMetrics], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["patch_id", "dice", "iou", "precision", "recall", "threshold"],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "patch_id": metric.patch_id,
                    "dice": f"{metric.dice:.6f}",
                    "iou": f"{metric.iou:.6f}",
                    "precision": f"{metric.precision:.6f}",
                    "recall": f"{metric.recall:.6f}",
                    "threshold": f"{metric.threshold:.1f}",
                }
            )


def _aggregate(metrics: list[PatchMetrics]) -> dict[str, float]:
    true_positive = sum(metric.true_positive for metric in metrics)
    false_positive = sum(metric.false_positive for metric in metrics)
    false_negative = sum(metric.false_negative for metric in metrics)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "dice": (2 * true_positive) / max(2 * true_positive + false_positive + false_negative, 1),
        "iou": true_positive / max(true_positive + false_positive + false_negative, 1),
        "precision": precision,
        "recall": recall,
    }


def _write_readme(output_path: Path, checkpoint: dict, aggregate: dict[str, float]) -> None:
    patch_ids = "\n".join(f"- `{patch_id}`" for patch_id in COMPLETED_RUN_TEST_PATCH_IDS)
    output_path.write_text(
        f"""# Test-patch visual evaluation

These artifacts evaluate the exact nine test patches used by the completed U-Net training run. A later export appended files to the test directory; this evaluation intentionally uses only the patch IDs listed below.

## Checkpoint

- Path: `ml/weights/best_model.pth`
- Encoder: `{checkpoint['encoder_name']}`
- Input channels: `{checkpoint['in_channels']}`
- Saved epoch: `{checkpoint['epoch']}`
- Saved validation loss: `{checkpoint['validation_loss']:.6f}`

## Rendering and threshold

- Probability threshold: `0.5` (`sigmoid(logits) >= 0.5`).
- The input panel is a false-color composite: red=NIR, green=Red, blue=Green.
- These channels are the model-normalized patch values: reflectance bands use `clip((x + 0.2) / 1.2, 0, 1)` and NDVI uses `clip((x + 1.0) / 2.0, 0, 1)`.
- The overlay panel uses the same false-color input with thresholded drought pixels in red.

## Exact completed-run test patches

{patch_ids}

## Reproduce

From `backend` with the active virtual environment:

```powershell
.\\.venv\\Scripts\\python.exe -m ml.evaluate_visuals
```

This writes one four-panel PNG per patch and `test_metrics.csv` in this directory without modifying the checkpoint or patch files.

## Aggregate micro metrics

| Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|
| {aggregate['dice']:.6f} | {aggregate['iou']:.6f} | {aggregate['precision']:.6f} | {aggregate['recall']:.6f} |
""",
        encoding="utf-8",
    )


def evaluate(checkpoint_path: Path = SEGMENTATION_MODEL_PATH, output_dir: Path = PREDICTIONS_DIR) -> dict[str, float]:
    """Evaluate and render the immutable test subset used by the completed run."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = NpyDroughtPatchDataset(DATASET_ROOT / "test" / "images", DATASET_ROOT / "test" / "labels")
    paths_by_id = {path.stem: path for path in dataset.image_paths}
    missing = [patch_id for patch_id in COMPLETED_RUN_TEST_PATCH_IDS if patch_id not in paths_by_id]
    if missing:
        raise FileNotFoundError(f"Completed-run test patches are missing: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("in_channels") != 7:
        raise ValueError(f"Expected a seven-channel checkpoint, received {checkpoint.get('in_channels')}.")
    model = build_unet(str(checkpoint.get("encoder_name", "resnet34"))).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[PatchMetrics] = []
    indices_by_id = {path.stem: index for index, path in enumerate(dataset.image_paths)}
    with torch.inference_mode():
        for patch_id in COMPLETED_RUN_TEST_PATCH_IDS:
            image_tensor, target_tensor = dataset[indices_by_id[patch_id]]
            probability = torch.sigmoid(model(image_tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            image = image_tensor.numpy()
            target = target_tensor[0].numpy() >= 0.5
            prediction = probability >= PREDICTION_THRESHOLD
            metrics.append(_metrics(patch_id, prediction, target))
            _save_panels(output_dir / f"{patch_id}.png", image, target, probability, prediction, patch_id)

    _write_csv(metrics, output_dir / "test_metrics.csv")
    aggregate = _aggregate(metrics)
    _write_readme(output_dir / "README.md", checkpoint, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SEGMENTATION_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=PREDICTIONS_DIR)
    arguments = parser.parse_args()
    aggregate = evaluate(arguments.checkpoint, arguments.output_dir)
    print(
        "Evaluated 9 patches. "
        f"dice={aggregate['dice']:.6f}, iou={aggregate['iou']:.6f}, "
        f"precision={aggregate['precision']:.6f}, recall={aggregate['recall']:.6f}"
    )


if __name__ == "__main__":
    main()

"""Audit drought-label coverage without changing dataset or model artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import DATASET_ROOT


def _region_from_patch_id(patch_id: str) -> str:
    region = patch_id.split("_", 1)[0]
    if not region:
        raise ValueError(f"Cannot infer region from patch ID: {patch_id}")
    return region


def audit_labels(labels_dir: Path) -> list[dict[str, str | int | float]]:
    """Return positive-pixel coverage for each label patch in sorted order."""
    rows: list[dict[str, str | int | float]] = []
    for label_path in sorted(labels_dir.glob("*.npy")):
        label = np.load(label_path)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        if label.ndim != 2:
            raise ValueError(f"Label patch {label_path.name} must have H,W dimensions.")
        total_pixels = int(label.size)
        positive_pixels = int(np.count_nonzero(label > 0))
        rows.append(
            {
                "patch_id": label_path.stem,
                "region": _region_from_patch_id(label_path.stem),
                "positive_pixels": positive_pixels,
                "total_pixels": total_pixels,
                "positive_fraction": positive_pixels / total_pixels,
            }
        )
    if not rows:
        raise ValueError(f"No .npy label patches found in {labels_dir}")
    return rows


def write_audit_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["patch_id", "region", "positive_pixels", "total_pixels", "positive_fraction"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "positive_fraction": f"{row['positive_fraction']:.9f}",
                }
            )


def write_summary(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["region"])].append(float(row["positive_fraction"]))
    lines = [
        "# Label coverage audit",
        "",
        "Each positive pixel is a stored label value greater than zero. The exporter creates that label with `VCI < 35.0%`, where VCI is computed from the composite NDVI and the per-pixel seasonal NDVI minimum and maximum.",
        "",
        "| Region | Patches | Minimum | Mean | Median | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in sorted(grouped):
        coverage = np.asarray(grouped[region])
        lines.append(
            f"| {region} | {len(coverage)} | {coverage.min():.2%} | {coverage.mean():.2%} | {np.median(coverage):.2%} | {coverage.max():.2%} |"
        )
    lines.extend(
        [
            "",
            "## Generation rule",
            "",
            "1. Sentinel-2 and Landsat observations are merged for May-August 2024.",
            "2. The six reflectance bands and NDVI are reduced with a median composite.",
            "3. `VCI = (composite_NDVI - seasonal_NDVI_min) / max(seasonal_NDVI_max - seasonal_NDVI_min, 0.0001) * 100`.",
            "4. A pixel is positive when `VCI < 35.0`; exported labels are then cast to `uint8` and thresholded with `label > 0`.",
            "",
            "This report is descriptive only. It does not retrain the model or modify any patch.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DATASET_ROOT / "test" / "labels")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "predictions" / "label_audit.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(__file__).resolve().parent / "weights" / "predictions" / "label_audit_summary.md",
    )
    arguments = parser.parse_args()
    rows = audit_labels(arguments.labels_dir)
    write_audit_csv(rows, arguments.output_csv)
    write_summary(rows, arguments.summary_path)
    print(f"Audited {len(rows)} label patches from {arguments.labels_dir}")


if __name__ == "__main__":
    main()
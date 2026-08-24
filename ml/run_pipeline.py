"""Export Earth Engine patches and train the drought-segmentation U-Net."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ml.export_gee_dataset import export_dataset
from ml.train import TrainingConfig, train_segmentation


def _timestamped(message: str) -> str:
    return f"{datetime.now(timezone.utc).isoformat()} {message}"


def _append_line(path: Path, line: str) -> None:
    """Append and flush directly to disk, independent of detached-process streams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{line}\n")
        output.flush()


def _safe_console_write(line: str, stream=None) -> None:
    """Best-effort console output; detached Windows handles may be invalid."""
    try:
        print(line, file=stream, flush=True)
    except OSError:
        pass


def _build_logger(log_path: Path):
    """Create a file-first progress logger that cannot fail because stdout is unavailable."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        line = _timestamped(message)
        _append_line(log_path, line)
        _safe_console_write(line)

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-patches", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--regions", nargs="+", default=["akmola", "karaganda"])
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Train from the existing data/patches split without downloading new tiles.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("ml/weights/pipeline-output-retry.log"),
        help="Path for flushed progress and final-metric logs.",
    )
    parser.add_argument(
        "--error-log-path",
        type=Path,
        default=Path("ml/weights/pipeline-error-retry.log"),
        help="Path for flushed traceback logs if the pipeline fails.",
    )
    arguments = parser.parse_args()
    log = _build_logger(arguments.log_path)

    arguments.error_log_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.error_log_path.write_text("", encoding="utf-8")
    try:
        if arguments.skip_export:
            log("Skipping export; training from existing patch files.")
        else:
            export = export_dataset(arguments.regions, arguments.year, arguments.max_patches)
            log(f"Export completed: {export.exported_patches} patches in {export.dataset_root}")

        def report_epoch(report: dict[str, float | int | bool]) -> None:
            log(
                "Epoch {epoch}/{total}: train_loss={train_loss:.6f}, "
                "validation_loss={validation_loss:.6f}, best_checkpoint={best_checkpoint}".format(
                    total=arguments.epochs,
                    **report,
                )
            )

        metrics = train_segmentation(
            TrainingConfig(epochs=arguments.epochs),
            progress_callback=report_epoch,
        )
        log(f"Training completed: {json.dumps(metrics, sort_keys=True)}")
    except BaseException:
        error_text = traceback.format_exc().rstrip()
        error_line = _timestamped(f"Pipeline failed:\n{error_text}")
        _append_line(arguments.error_log_path, error_line)
        _safe_console_write(error_line, stream=sys.stderr)
        raise


if __name__ == "__main__":
    main()

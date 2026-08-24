"""Local checkpoint inference for remote-sensing drought segmentation patches."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from app.config import PATCH_CHANNELS, PATCH_SIZE, SEGMENTATION_MODEL_PATH
from ml.dataset import normalize_patch
from ml.train import build_unet


class SegmentationModelUnavailableError(RuntimeError):
    """Raised when inference is requested before a segmentation checkpoint exists."""


def validate_patch(array: np.ndarray) -> np.ndarray:
    """Validate the raw seven-channel patch shape and numeric values."""
    expected_shape = (len(PATCH_CHANNELS), PATCH_SIZE, PATCH_SIZE)
    if array.shape != expected_shape:
        raise ValueError(f"Expected patch shape {expected_shape}, received {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Patch values must be numeric.")
    return np.asarray(array, dtype=np.float32)


def load_uploaded_npy(contents: bytes) -> np.ndarray:
    """Read a non-pickled uploaded numpy patch without writing untrusted data to disk."""
    try:
        array = np.load(io.BytesIO(contents), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Upload must be a valid non-pickled .npy array.") from error
    return validate_patch(np.asarray(array))


def infer_patch(patch: np.ndarray, checkpoint_path: Path = SEGMENTATION_MODEL_PATH) -> tuple[np.ndarray, float]:
    """Return a 256x256 drought-probability mask and its drought-pixel fraction."""
    if not checkpoint_path.exists():
        raise SegmentationModelUnavailableError(
            "Segmentation model is unavailable. Generate patches and train `backend/ml/train.py` first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder_name = str(checkpoint.get("encoder_name", "resnet34"))
    model = build_unet(encoder_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized_patch = normalize_patch(validate_patch(patch))
    tensor = torch.from_numpy(np.ascontiguousarray(normalized_patch)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy().astype(np.float32)
    return probability, float((probability >= 0.5).mean())

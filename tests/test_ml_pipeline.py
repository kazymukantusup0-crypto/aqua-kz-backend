from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from app.api import routes_segmentation
from app.config import PATCH_SIZE
from app.main import app
from app.services.segmentation_service import SegmentationModelUnavailableError
from ml.dataset import NpyDroughtPatchDataset, normalize_patch


def test_normalize_patch_has_seven_channels_and_safe_range() -> None:
    patch = np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    patch[6] = -1.0
    normalized = normalize_patch(patch)
    assert normalized.shape == patch.shape
    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_npy_dataset_loads_paired_patch(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    np.save(images / "sample.npy", np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
    np.save(labels / "sample.npy", np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8))
    dataset = NpyDroughtPatchDataset(images, labels)
    image, label = dataset[0]
    assert tuple(image.shape) == (7, PATCH_SIZE, PATCH_SIZE)
    assert tuple(label.shape) == (1, PATCH_SIZE, PATCH_SIZE)


def test_uploaded_tile_returns_clear_model_not_ready_response(monkeypatch) -> None:
    def unavailable_model(*_args, **_kwargs):
        raise SegmentationModelUnavailableError("Segmentation model is unavailable for this test.")

    monkeypatch.setattr(routes_segmentation, "infer_patch", unavailable_model)
    buffer = BytesIO()
    np.save(buffer, np.zeros((7, PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
    client = TestClient(app)
    response = client.post(
        "/api/ml/infer/upload",
        files={"tile": ("tile.npy", buffer.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 409
    assert "Segmentation model is unavailable" in response.json()["detail"]

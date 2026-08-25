import sqlite3
import struct
import zlib
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from app.api import routes_segmentation
from app.main import app
from app.services import inference_run_service


def read_png(content: bytes) -> tuple[int, int, int, bytes]:
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    width = height = color_type = 0
    compressed = b""
    while position < len(content):
        length = struct.unpack(">I", content[position : position + 4])[0]
        chunk_type = content[position + 4 : position + 8]
        chunk = content[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
            assert bit_depth == 8
        elif chunk_type == b"IDAT":
            compressed += chunk
        elif chunk_type == b"IEND":
            break
    return width, height, color_type, zlib.decompress(compressed)


def configure_storage(monkeypatch, tmp_path):
    database_path = tmp_path / "aqua_kz.db"
    predictions_path = tmp_path / "predictions"
    monkeypatch.setattr(inference_run_service, "DB_PATH", database_path)
    monkeypatch.setattr(inference_run_service, "PREDICTIONS_DIR", predictions_path)
    return database_path, predictions_path


def test_storage_initialization_creates_database_and_table(monkeypatch, tmp_path) -> None:
    database_path, predictions_path = configure_storage(monkeypatch, tmp_path)

    inference_run_service.initialize_storage()

    assert database_path.is_file()
    assert predictions_path.is_dir()
    with sqlite3.connect(database_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'inference_runs'"
        ).fetchone()
    assert table == ("inference_runs",)


def test_save_and_retrieve_run_round_trips_binary_mask(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    probability_mask = np.array([[0.1, 0.5], [0.9, 0.2]], dtype=np.float32)

    run = inference_run_service.save_inference_run(
        probability_mask,
        drought_fraction=0.5,
        threshold=0.5,
        source_type="coordinates",
        latitude=48.0,
        longitude=67.0,
        year=2024,
    )
    retrieved = inference_run_service.get_inference_run(run["id"])

    assert retrieved == run
    assert run["drought_pixel_count"] == 2
    assert run["total_pixel_count"] == 4
    np.testing.assert_array_equal(inference_run_service.load_saved_mask(run), [[0, 1], [1, 0]])


def test_list_runs_is_newest_first(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    first = inference_run_service.save_inference_run(np.zeros((2, 2)), 0, 0.5, "upload", "first.npy")
    second = inference_run_service.save_inference_run(np.ones((2, 2)), 1, 0.5, "upload", "second.npy")

    runs = inference_run_service.list_inference_runs()

    assert [run["id"] for run in runs] == [second["id"], first["id"]]


def test_mask_download_and_404_cases(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    run = inference_run_service.save_inference_run(np.ones((2, 2)), 1, 0.5, "upload", "tile.npy")
    client = TestClient(app)

    response = client.get(f"/api/ml/runs/{run['id']}/mask")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == f'attachment; filename="{run["id"]}_mask.npy"'
    np.testing.assert_array_equal(np.load(BytesIO(response.content), allow_pickle=False), np.ones((2, 2), dtype=np.uint8))

    missing_run = client.get("/api/ml/runs/not-a-run")
    missing_mask = client.get("/api/ml/runs/not-a-run/mask")
    assert missing_run.status_code == 404
    assert missing_mask.status_code == 404

    (tmp_path / "predictions" / f"{run['id']}_mask.npy").unlink()
    missing_file = client.get(f"/api/ml/runs/{run['id']}/mask")
    assert missing_file.status_code == 404
    assert "saved inference mask" in missing_file.json()["detail"]


def test_save_creates_valid_preview_and_overlay_artifacts(monkeypatch, tmp_path) -> None:
    _, predictions_path = configure_storage(monkeypatch, tmp_path)
    binary_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    run = inference_run_service.save_inference_run(binary_mask, 0.5, 0.5, "upload", "tile.npy")
    preview = (predictions_path / f"{run['id']}_mask.png").read_bytes()
    overlay = (predictions_path / f"{run['id']}_overlay.png").read_bytes()
    preview_width, preview_height, preview_type, preview_data = read_png(preview)
    overlay_width, overlay_height, overlay_type, overlay_data = read_png(overlay)

    assert (preview_width, preview_height, preview_type) == (2, 2, 0)
    assert preview_data == b"\x00\x00\xff\x00\xff\x00"
    assert (overlay_width, overlay_height, overlay_type) == (2, 2, 6)
    assert overlay_data[:9] == b"\x00" + bytes([0, 0, 0, 0, 220, 38, 38, 165])
    assert overlay_data[9:] == b"\x00" + bytes([220, 38, 38, 165, 0, 0, 0, 0])


def test_preview_and_overlay_endpoints_return_png_and_missing_artifacts_404(monkeypatch, tmp_path) -> None:
    _, predictions_path = configure_storage(monkeypatch, tmp_path)
    run = inference_run_service.save_inference_run(np.ones((2, 2)), 1, 0.5, "upload", "tile.npy")
    client = TestClient(app)

    preview = client.get(f"/api/ml/runs/{run['id']}/preview")
    overlay = client.get(f"/api/ml/runs/{run['id']}/overlay")
    assert preview.status_code == overlay.status_code == 200
    assert preview.headers["content-type"] == overlay.headers["content-type"] == "image/png"

    (predictions_path / f"{run['id']}_mask.png").unlink()
    (predictions_path / f"{run['id']}_overlay.png").unlink()
    assert client.get(f"/api/ml/runs/{run['id']}/preview").status_code == 404
    assert client.get(f"/api/ml/runs/{run['id']}/overlay").status_code == 404


def test_persistent_upload_inference_reuses_inference_flow(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(routes_segmentation, "load_uploaded_npy", lambda _contents: np.zeros((7, 2, 2)))
    monkeypatch.setattr(
        routes_segmentation,
        "infer_patch",
        lambda _patch: (np.array([[0.2, 0.8], [0.9, 0.1]]), 0.5),
    )
    client = TestClient(app)

    response = client.post(
        "/api/ml/runs/infer/upload",
        files={"tile": ("folder/tile.npy", b"ignored", "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_type"] == "upload"
    assert payload["upload_filename"] == "tile.npy"
    assert "mask" not in payload
    assert client.get(payload["mask_url"]).status_code == 200


def test_persistent_coordinate_inference_reuses_inference_flow(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(routes_segmentation, "fetch_patch_for_coordinates", lambda *_args: np.zeros((7, 2, 2)))
    monkeypatch.setattr(routes_segmentation, "infer_patch", lambda _patch: (np.ones((2, 2)), 1.0))
    client = TestClient(app)

    response = client.post(
        "/api/ml/runs/infer/coordinates",
        json={"latitude": 48, "longitude": 67, "year": 2024},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_type"] == "coordinates"
    assert payload["latitude"] == 48
    assert payload["longitude"] == 67
    assert payload["year"] == 2024
    assert "mask" not in payload
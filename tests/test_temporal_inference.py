from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from app.api import routes_monitoring, routes_segmentation
from app.main import app
from app.services import inference_run_service
from ml.export_gee_dataset import SatelliteImageryUnavailableError, scene_date_range


def configure_storage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    monkeypatch.setattr(inference_run_service, "PREDICTIONS_DIR", tmp_path / "predictions")


def test_monthly_scene_date_ranges_include_year_rollover() -> None:
    assert scene_date_range(2024, 1) == (date(2024, 1, 1), date(2024, 2, 1))
    assert scene_date_range(2024, 6) == (date(2024, 6, 1), date(2024, 7, 1))
    assert scene_date_range(2024, 12) == (date(2024, 12, 1), date(2025, 1, 1))


def test_coordinate_inference_accepts_optional_month_and_persists_provenance(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        routes_segmentation,
        "fetch_patch_for_coordinates",
        lambda *args: (calls.append(args) or np.zeros((7, 2, 2))),
    )
    monkeypatch.setattr(routes_segmentation, "infer_patch", lambda _patch: (np.ones((2, 2)), 1.0))
    client = TestClient(app)

    response = client.post(
        "/api/ml/runs/infer/coordinates",
        json={"latitude": 49.8, "longitude": 73.1, "year": 2024, "month": 6},
    )

    assert response.status_code == 200
    assert calls == [(49.8, 73.1, 2024, 6)]
    payload = response.json()
    assert payload["requested_year"] == 2024
    assert payload["requested_month"] == 6
    assert payload["scene_start_date"] == "2024-06-01"
    assert payload["scene_end_date"] == "2024-07-01"


def test_monitoring_analyze_and_backfill_pass_months(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        routes_monitoring,
        "fetch_patch_for_coordinates",
        lambda *args: (calls.append(args) or np.zeros((7, 2, 2))),
    )
    monkeypatch.setattr(routes_monitoring, "infer_patch", lambda _patch: (np.zeros((2, 2)), 0.0))
    client = TestClient(app)
    location = client.post(
        "/api/monitoring/locations",
        json={"name": "Karaganda", "latitude": 49.8, "longitude": 73.1},
    ).json()

    analyzed = client.post(
        f"/api/monitoring/locations/{location['id']}/analyze",
        json={"year": 2024, "month": 6},
    )
    backfill = client.post(
        f"/api/monitoring/locations/{location['id']}/backfill",
        json={"start_date": "2024-07-01", "end_date": "2024-08-01"},
    )

    assert analyzed.status_code == 200
    assert backfill.status_code == 200
    assert [call[3] for call in calls] == [6, 7, 8]
    assert analyzed.json()["inference_run"]["requested_month"] == 6


def test_missing_monthly_imagery_is_a_not_found(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    monkeypatch.setattr(
        routes_segmentation,
        "fetch_patch_for_coordinates",
        lambda *_args: (_ for _ in ()).throw(
            SatelliteImageryUnavailableError(
                "No usable satellite imagery found for 2024-06 at the requested location after cloud filtering."
            )
        ),
    )
    response = TestClient(app).post(
        "/api/ml/infer/coordinates",
        json={"latitude": 49.8, "longitude": 73.1, "year": 2024, "month": 6},
    )
    assert response.status_code == 404
    assert "2024-06" in response.json()["detail"]

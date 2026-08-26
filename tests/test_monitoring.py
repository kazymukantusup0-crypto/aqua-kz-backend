from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from app.api import routes_monitoring
from app.main import app
from app.services import inference_run_service, monitoring_service
from app.services.drought_risk_service import assess_drought_risk
from app.services.nasa_power_service import MonthlyWeather
from app.services.vegetation_service import VegetationIndicators


def configure_storage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    monkeypatch.setattr(inference_run_service, "PREDICTIONS_DIR", tmp_path / "predictions")


def test_risk_engine_renormalizes_and_reports_missing_inputs() -> None:
    result = assess_drought_risk(drought_fraction=0.8, precipitation_anomaly=-1.0)

    assert 0.9 < result.risk_score <= 1.0
    assert result.risk_level == "critical"
    assert result.data_quality == "partial"
    assert "vegetation_stress" in result.missing_inputs
    assert "unavailable inputs" in result.explanation


def test_partial_karaganda_case_keeps_segmentation_weight_floor() -> None:
    result = assess_drought_risk(
        drought_fraction=0.483,
        rainfall_stress=0.428,
        heat_stress=0.0,
    )
    assert result.risk_score == 0.3429
    assert result.risk_level == "medium"
    assert result.data_quality == "partial"


def test_risk_engine_thresholds_and_segmentation_only() -> None:
    assert assess_drought_risk(drought_fraction=0.1).risk_level == "low"
    assert assess_drought_risk(drought_fraction=0.5).risk_level == "medium"
    assert assess_drought_risk(drought_fraction=0.75).risk_level == "high"
    critical = assess_drought_risk(drought_fraction=0.9)
    assert critical.risk_level == "critical"
    assert critical.data_quality == "segmentation_only"


def test_trend_states_are_deterministic() -> None:
    previous = {"risk_score": 0.5, "drought_fraction": 0.5}
    assert monitoring_service.calculate_trend({"risk_score": 0.6, "drought_fraction": 0.5}, previous) == "worsening"
    assert monitoring_service.calculate_trend({"risk_score": 0.4, "drought_fraction": 0.5}, previous) == "improving"
    assert monitoring_service.calculate_trend({"risk_score": 0.52, "drought_fraction": 0.52}, previous) == "stable"
    assert monitoring_service.calculate_trend(previous, None) == "insufficient_data"


def test_location_crud_duplicate_and_validation(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = {"name": "Karaganda demo point", "latitude": 49.8, "longitude": 73.1, "region": "Karaganda"}

    created = client.post("/api/monitoring/locations", json=payload)
    assert created.status_code == 201
    location = created.json()
    location_id = location["id"]
    assert client.post("/api/monitoring/locations", json=payload).status_code == 409
    assert client.post("/api/monitoring/locations", json={**payload, "latitude": 100}).status_code == 422
    assert client.get(f"/api/monitoring/locations/{location_id}").status_code == 200
    assert client.patch(f"/api/monitoring/locations/{location_id}", json={"name": "Renamed point"}).json()["name"] == "Renamed point"
    deleted = client.delete(f"/api/monitoring/locations/{location_id}")
    assert deleted.status_code == 200 and deleted.json()["active"] is False
    assert client.get("/api/monitoring/locations/not-found").status_code == 404


def mock_inference(monkeypatch) -> None:
    monkeypatch.setattr(routes_monitoring, "fetch_patch_for_coordinates", lambda *_args: np.zeros((7, 2, 2)))
    monkeypatch.setattr(routes_monitoring, "infer_patch", lambda _patch: (np.array([[0.1, 0.8], [0.2, 0.9]]), 0.5))


def create_location(client: TestClient) -> dict:
    return client.post(
        "/api/monitoring/locations",
        json={"name": "Karaganda demo point", "latitude": 49.8, "longitude": 73.1, "region": "Karaganda"},
    ).json()


def test_analyze_latest_timeline_and_outlook(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    client = TestClient(app)
    location = create_location(client)
    location_id = location["id"]

    analyzed = client.post(f"/api/monitoring/locations/{location_id}/analyze", json={"year": 2024, "month": 5})
    assert analyzed.status_code == 200
    result = analyzed.json()
    assert result["data_quality"] == "segmentation_only"
    assert result["observation"]["trend"] == "insufficient_data"
    assert result["preview_url"].endswith("/preview")
    assert "mask" not in result
    duplicate = client.post(f"/api/monitoring/locations/{location_id}/analyze", json={"year": 2024, "month": 5})
    assert duplicate.status_code == 409

    latest = client.get(f"/api/monitoring/locations/{location_id}/latest")
    timeline = client.get(f"/api/monitoring/locations/{location_id}/timeline?include_runs=true")
    outlook = client.get(f"/api/monitoring/locations/{location_id}/outlook")
    assert latest.status_code == 200
    assert timeline.status_code == 200 and timeline.json()["summary"]["observation_count"] == 1
    assert timeline.json()["observations"][0]["inference_run"]["id"] == result["inference_run"]["id"]
    assert outlook.json()["forecast_available"] is False


def test_analyze_enrichment_populates_weather_and_partial_quality(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    monkeypatch.setattr(
        routes_monitoring,
        "fetch_monthly_weather",
        lambda *_args: MonthlyWeather(32.0, 24.5, 47.0, 22.4, -32 / 47, 2.1, 32 / 47, 0.35),
    )
    client = TestClient(app)
    location_id = create_location(client)["id"]
    response = client.post(
        f"/api/monitoring/locations/{location_id}/analyze",
        json={"year": 2024, "month": 5, "enrich": True},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["data_quality"] == "partial"
    assert result["observation"]["precipitation_mm"] == 32.0
    assert result["observation"]["temperature_anomaly"] == 2.1
    assert "below the historical baseline" in result["explanation"]
    assert "Temperature was about +2.1 C" in result["explanation"]


def test_analyze_enrichment_failure_falls_back_to_segmentation_only(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    monkeypatch.setattr(routes_monitoring, "fetch_monthly_weather", lambda *_args: (_ for _ in ()).throw(TimeoutError()))
    client = TestClient(app)
    location_id = create_location(client)["id"]
    response = client.post(
        f"/api/monitoring/locations/{location_id}/analyze",
        json={"year": 2024, "month": 5, "enrich": True},
    )
    assert response.status_code == 200
    assert response.json()["data_quality"] == "segmentation_only"


def test_analyze_satellite_enrichment_failure_preserves_weather(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    monkeypatch.setattr(
        routes_monitoring,
        "fetch_monthly_weather",
        lambda *_args: MonthlyWeather(32.0, 24.5, 47.0, 22.4, -32 / 47, 2.1, 32 / 47, 0.35),
    )
    monkeypatch.setattr(
        routes_monitoring,
        "enrich_patch",
        lambda *_args: (_ for _ in ()).throw(routes_monitoring.VegetationEnrichmentError()),
    )
    client = TestClient(app)
    location_id = create_location(client)["id"]
    response = client.post(
        f"/api/monitoring/locations/{location_id}/analyze",
        json={"year": 2024, "month": 5, "enrich": True},
    )
    assert response.status_code == 200
    assert response.json()["data_quality"] == "partial"
    assert response.json()["observation"]["vegetation_stress"] is None


def test_analyze_satellite_enrichment_populates_indices_and_stress(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    monkeypatch.setattr(
        routes_monitoring,
        "enrich_patch",
        lambda *_args: VegetationIndicators(
            0.30, 0.20, 0.38, 0.25, -0.08, -0.05, 0.4
        ),
    )
    client = TestClient(app)
    location_id = create_location(client)["id"]
    response = client.post(
        f"/api/monitoring/locations/{location_id}/analyze",
        json={"year": 2024, "month": 5, "enrich": True},
    )
    assert response.status_code == 200
    observation = response.json()["observation"]
    assert observation["ndvi"] == 0.3
    assert observation["ndmi"] == 0.2
    assert observation["ndvi_anomaly"] == -0.08
    assert observation["vegetation_stress"] == 0.4
    assert observation["data_quality"] == "partial"


def test_analyze_without_enrichment_does_not_call_nasa_power(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    monkeypatch.setattr(routes_monitoring, "fetch_monthly_weather", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    client = TestClient(app)
    location_id = create_location(client)["id"]
    response = client.post(
        f"/api/monitoring/locations/{location_id}/analyze",
        json={"year": 2024, "month": 5, "enrich": False},
    )
    assert response.status_code == 200
    assert response.json()["data_quality"] == "segmentation_only"


def test_timeline_order_trend_and_backfill_deduplication(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    mock_inference(monkeypatch)
    client = TestClient(app)
    location_id = create_location(client)["id"]
    client.post(f"/api/monitoring/locations/{location_id}/analyze", json={"year": 2024, "month": 1})
    client.post(f"/api/monitoring/locations/{location_id}/analyze", json={"year": 2024, "month": 2})
    timeline = client.get(f"/api/monitoring/locations/{location_id}/timeline").json()
    assert [item["month"] for item in timeline["observations"]] == [1, 2]
    assert timeline["summary"]["trend"] == "stable"
    assert client.post(
        f"/api/monitoring/locations/{location_id}/backfill",
        json={"start_date": "2024-01-15", "end_date": "2024-03-20"},
    ).json()["skipped"] == 2
    assert client.post(
        f"/api/monitoring/locations/{location_id}/backfill",
        json={"start_date": "2024-01-01", "end_date": "2026-01-01"},
    ).status_code == 422
    assert client.post(
        f"/api/monitoring/locations/{location_id}/backfill",
        json={"start_date": "2024-05-01", "end_date": "2024-04-01"},
    ).status_code == 422


def test_analyze_preserves_engine_and_model_errors(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    client = TestClient(app)
    location_id = create_location(client)["id"]
    monkeypatch.setattr(routes_monitoring, "fetch_patch_for_coordinates", lambda *_args: (_ for _ in ()).throw(
        routes_monitoring.EarthEngineConfigurationError("Earth Engine unavailable")
    ))
    response = client.post(f"/api/monitoring/locations/{location_id}/analyze", json={"year": 2024, "month": 5})
    assert response.status_code == 503
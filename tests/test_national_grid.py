from argparse import Namespace

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import inference_run_service, monitoring_service, national_grid_service
from app.services.kazakhstan_grid import configured_grid_cells, point_in_kazakhstan
from scripts import prepare_national_grid


def configure_storage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    monkeypatch.setattr(inference_run_service, "PREDICTIONS_DIR", tmp_path / "predictions")


def test_boundary_filter_and_grid_ids_are_stable() -> None:
    cells = configured_grid_cells()
    assert len(cells) == 329
    assert cells == configured_grid_cells()
    assert all(point_in_kazakhstan(cell["latitude"], cell["longitude"]) for cell in cells)
    assert all(cell["cell_id"].startswith("kz-grid-v1-") for cell in cells)
    assert point_in_kazakhstan(49.5, 69.5)
    assert not point_in_kazakhstan(40.5, 46.5)


def test_grid_storage_is_additive_and_keeps_user_point(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    karaganda = monitoring_service.create_location("Karaganda demo point", 49.8, 73.1, "Karaganda")
    first = national_grid_service.ensure_grid_cells()
    second = national_grid_service.ensure_grid_cells()
    assert len(first) == len(second) == 329
    assert monitoring_service.get_location(karaganda["id"])["name"] == "Karaganda demo point"
    assert [location["id"] for location in monitoring_service.list_locations()] == [karaganda["id"]]


def test_national_grid_endpoint_explicitly_reports_missing_and_summary_counts(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    cells = national_grid_service.ensure_grid_cells()
    monitoring_service.save_observation({
        "location_id": cells[2]["location_id"], "observed_at": "2024-09-01", "inference_run_id": "grid-test-run",
        "year": 2024, "month": 9, "drought_fraction": 0.8, "drought_pixel_count": 8,
        "total_pixel_count": 10, "segmentation_stress": 0.8, "risk_score": 0.8, "risk_level": "high",
        "trend": "worsening", "explanation": "Persisted screening observation.", "data_quality": "segmentation_only",
        "data_source": "test",
    })
    national_grid_service.record_status("kz-grid-v1", cells[0]["cell_id"], 2024, 9, "failed", "provider unavailable")
    national_grid_service.record_status("kz-grid-v1", cells[1]["cell_id"], 2024, 9, "stale", "reprocess required")
    client = TestClient(app)
    response = client.get("/api/monitoring/national-grid?year=2024&month=9")
    assert response.status_code == 200
    payload = {item["cell_id"]: item for item in response.json()}
    assert payload[cells[0]["cell_id"]]["status"] == "failed"
    assert payload[cells[2]["cell_id"]]["status"] == "ready"
    assert payload[cells[2]["cell_id"]]["risk_score"] == 0.8
    missing = next(item for item in payload.values() if item["status"] == "missing")
    assert missing["risk_score"] is None and missing["risk_level"] is None
    summary = client.get("/api/monitoring/national-grid/summary?year=2024&month=9").json()
    assert summary["total_configured_cells"] == 329
    assert summary["ready"] == 1 and summary["failed"] == 1 and summary["stale"] == 1 and summary["missing"] == 326
    assert summary["risk_counts"] == {"low": 0, "medium": 0, "high": 1, "critical": 0}


def test_screening_segmentation_only_cell_uses_preliminary_signal_display(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    cells = national_grid_service.ensure_grid_cells()
    monitoring_service.save_observation({
        "location_id": cells[3]["location_id"], "observed_at": "2024-09-03", "inference_run_id": "grid-segmentation-only",
        "year": 2024, "month": 9, "drought_fraction": 1.0, "drought_pixel_count": 256,
        "total_pixel_count": 256, "segmentation_stress": 1.0, "risk_score": 1.0, "risk_level": "critical",
        "trend": "insufficient_data", "explanation": "Segmentation-only screening signal.", "data_quality": "segmentation_only",
        "data_source": "test",
    })
    client = TestClient(app)
    payload = {item["cell_id"]: item for item in client.get("/api/monitoring/national-grid?year=2024&month=9").json()}
    cell = payload[cells[3]["cell_id"]]
    assert cell["risk_score"] == 1.0
    assert cell["risk_level"] == "critical"
    assert cell["assessment_mode"] == "preliminary_segmentation_signal"
    assert "ограниченная" in cell["confidence_label"].lower()
    assert "сегментация" in cell["confidence_label"].lower()


def test_partial_data_cell_remains_multi_signal_risk_display(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    cells = national_grid_service.ensure_grid_cells()
    monitoring_service.save_observation({
        "location_id": cells[4]["location_id"], "observed_at": "2024-09-04", "inference_run_id": "grid-partial",
        "year": 2024, "month": 9, "drought_fraction": 0.5, "drought_pixel_count": 128,
        "total_pixel_count": 256, "segmentation_stress": 0.45, "risk_score": 0.6, "risk_level": "medium",
        "trend": "worsening", "explanation": "Partial-data risk estimate.", "data_quality": "partial",
        "data_source": "test",
    })
    client = TestClient(app)
    payload = {item["cell_id"]: item for item in client.get("/api/monitoring/national-grid?year=2024&month=9").json()}
    cell = payload[cells[4]["cell_id"]]
    assert cell["data_quality"] == "partial"
    assert cell["assessment_mode"] == "multi_signal_risk"
    assert "частичная" in cell["confidence_label"].lower() or "несколько" in cell["confidence_label"].lower()


def test_missing_screening_cells_remain_no_data_and_never_low_risk(monkeypatch, tmp_path) -> None:
    configure_storage(monkeypatch, tmp_path)
    national_grid_service.ensure_grid_cells()
    client = TestClient(app)
    payload = {item["cell_id"]: item for item in client.get("/api/monitoring/national-grid?year=2024&month=9").json()}
    missing = next(item for item in payload.values() if item["status"] == "missing")
    assert missing["risk_score"] is None
    assert missing["risk_level"] is None
    assert missing["assessment_mode"] == "no_data"
    assert missing["confidence_label"] == "Нет данных"
    assert all(item["status"] != "missing" or item["risk_level"] is None for item in payload.values())


def args(**overrides) -> Namespace:
    values = {"year": 2024, "month": 9, "grid_version": "kz-grid-v1", "limit": None, "dry_run": False, "resume": True, "cell_id": [], "enrich": False, "force": False}
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_does_not_persist_or_call_external_work(monkeypatch, capsys) -> None:
    cells = [{"cell_id": "kz-grid-v1-1-1", "latitude": 49.5, "longitude": 69.5}]
    monkeypatch.setattr(prepare_national_grid, "configured_grid_cells", lambda *_args: cells)
    monkeypatch.setattr(prepare_national_grid, "grid_plan_summary", lambda *_args: {"inside_kazakhstan_count": 1, "candidate_count": 1})
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "ensure_grid_cells", lambda *_args: (_ for _ in ()).throw(AssertionError("write")))
    monkeypatch.setattr(prepare_national_grid, "_perform_analysis", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("external")))
    assert prepare_national_grid.run(args(dry_run=True)) == 0
    assert "zero database writes" in capsys.readouterr().out


def test_resume_skips_a_ready_cell(monkeypatch) -> None:
    cells = [{"cell_id": "kz-grid-v1-1-1", "latitude": 49.5, "longitude": 69.5}]
    persisted = [{**cells[0], "location_id": "location-1"}]
    monkeypatch.setattr(prepare_national_grid, "configured_grid_cells", lambda *_args: cells)
    monkeypatch.setattr(prepare_national_grid, "grid_plan_summary", lambda *_args: {"inside_kazakhstan_count": 1, "candidate_count": 1})
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "ensure_grid_cells", lambda *_args: persisted)
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "grid_results", lambda *_args, **_kwargs: [{"cell_id": "kz-grid-v1-1-1", "status": "ready"}])
    monkeypatch.setattr(prepare_national_grid, "_perform_analysis", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ready cells must be skipped")))
    assert prepare_national_grid.run(args()) == 0


def test_resume_skips_ready_and_failure_continues_to_next_cell(monkeypatch) -> None:
    cells = [
        {"cell_id": "kz-grid-v1-1-1", "latitude": 49.5, "longitude": 69.5},
        {"cell_id": "kz-grid-v1-1-2", "latitude": 49.5, "longitude": 70.5},
    ]
    persisted = [{**cell, "location_id": f"location-{index}"} for index, cell in enumerate(cells)]
    monkeypatch.setattr(prepare_national_grid, "configured_grid_cells", lambda *_args: cells)
    monkeypatch.setattr(prepare_national_grid, "grid_plan_summary", lambda *_args: {"inside_kazakhstan_count": 2, "candidate_count": 2})
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "ensure_grid_cells", lambda *_args: persisted)
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "grid_results", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(prepare_national_grid.monitoring_service, "get_location", lambda location_id: {"id": location_id})
    statuses = []
    monkeypatch.setattr(prepare_national_grid.national_grid_service, "record_status", lambda *_args, **_kwargs: statuses.append(_args[4]))
    calls = []
    def analysis(location, *_args, **_kwargs):
        calls.append(location["id"])
        if len(calls) == 1:
            raise HTTPException(status_code=503, detail="temporary Earth Engine error")
    monkeypatch.setattr(prepare_national_grid, "_perform_analysis", analysis)
    monkeypatch.setattr(prepare_national_grid.time, "sleep", lambda _seconds: None)
    assert prepare_national_grid.run(args()) == 1
    assert calls == ["location-0", "location-1"]
    assert statuses == ["failed", "ready"]

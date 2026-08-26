import numpy as np
from pytest import approx

from app.services import inference_run_service
from app.services import vegetation_service


def test_patch_indices_use_shared_normalized_bands() -> None:
    patch = np.zeros((7, 2, 2), dtype=float)
    patch[2] = 0.2  # Red
    patch[3] = 0.6  # NIR
    patch[4] = 0.3  # SWIR1
    ndvi, ndmi = vegetation_service.calculate_patch_indices(patch)
    assert ndvi == approx(0.5)
    assert ndmi == approx(0.3333333333333333)


def test_vegetation_stress_boundaries() -> None:
    assert vegetation_service.vegetation_stress_from_anomaly(0.0) == 0.0
    assert vegetation_service.vegetation_stress_from_anomaly(0.1) == 0.0
    assert vegetation_service.vegetation_stress_from_anomaly(-0.1) == 0.5
    assert vegetation_service.vegetation_stress_from_anomaly(-0.2) == 1.0
    assert vegetation_service.vegetation_stress_from_anomaly(-0.4) == 1.0
    assert vegetation_service.vegetation_stress_from_anomaly(None) is None


def test_baseline_anomaly_and_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    calls = []

    def fake_fetch(_latitude, _longitude, year, _month):
        calls.append(year)
        patch = np.zeros((7, 1, 1), dtype=float)
        patch[2, 0, 0] = 0.2
        patch[3, 0, 0] = 0.4 + (year - 2014) * 0.01
        patch[4, 0, 0] = 0.2
        return patch

    monkeypatch.setattr(vegetation_service, "fetch_patch_for_coordinates", fake_fetch)
    current = np.zeros((7, 1, 1), dtype=float)
    current[2, 0, 0] = 0.2
    current[3, 0, 0] = 0.4
    current[4, 0, 0] = 0.2
    result = vegetation_service.enrich_patch(current, 49.8, 73.1, 2024, 5)
    assert calls == list(range(2014, 2024))
    assert result.ndvi_baseline == sum((0.4 + (year - 2014) * 0.01 - 0.2) / (0.4 + (year - 2014) * 0.01 + 0.2) for year in range(2014, 2024)) / 10
    assert result.ndvi_anomaly == result.ndvi - result.ndvi_baseline
    assert result.vegetation_stress is not None
    vegetation_service.enrich_patch(current, 49.8, 73.1, 2024, 5)
    assert calls == list(range(2014, 2024))
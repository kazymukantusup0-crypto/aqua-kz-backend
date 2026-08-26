import json
from datetime import date

from app.services import inference_run_service
from app.services import nasa_power_service


class FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def payload(values: dict[str, dict[str, float]]) -> dict:
    return {"properties": {"parameter": {"PRECTOTCORR": values["precipitation"], "T2M": values["temperature"]}}}


def test_monthly_weather_parses_daily_values_and_ignores_fill_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    calls = []
    current = {"20240501": 2.0, "20240502": -999.0, "20240503": 4.0}
    temperatures = {"20240501": 20.0, "20240502": -999.0, "20240503": 24.0}
    baseline = {}
    for year in range(2014, 2024):
        baseline[f"{year}0501"] = 1.0
        baseline[f"{year}0502"] = 3.0
    baseline_temperatures = {day: 20.0 for day in baseline}

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if "start=20240501" in request.full_url:
            return FakeResponse(payload({"precipitation": current, "temperature": temperatures}))
        return FakeResponse(payload({"precipitation": baseline, "temperature": baseline_temperatures}))

    monkeypatch.setattr(nasa_power_service, "urlopen", fake_urlopen)
    result = nasa_power_service.fetch_monthly_weather(49.8, 73.1, 2024, 5)
    assert result.precipitation_mm == 6.0
    assert result.temperature_c == 22.0
    assert result.baseline_precipitation_mm == 4.0
    assert result.baseline_temperature_c == 20.0
    assert result.precipitation_anomaly == 0.5
    assert result.temperature_anomaly == 2.0
    assert result.rainfall_stress == 0.0
    assert result.heat_stress == 2 / 6
    assert len(calls) == 2


def test_stress_normalization_boundaries() -> None:
    assert nasa_power_service._rainfall_stress(0.0) == 0.0
    assert nasa_power_service._rainfall_stress(-1.0) == 1.0
    assert nasa_power_service._rainfall_stress(-2.0) == 1.0
    assert nasa_power_service._heat_stress(0.0) == 0.0
    assert nasa_power_service._heat_stress(6.0) == 1.0
    assert nasa_power_service._heat_stress(10.0) == 1.0


def test_baseline_cache_avoids_redundant_request(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(inference_run_service, "DB_PATH", tmp_path / "aqua_kz.db")
    calls = []
    values = {f"{year}0501": 2.0 for year in range(2014, 2024)}
    temperatures = {day: 20.0 for day in values}

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if "start=20240501" in request.full_url:
            return FakeResponse(payload({"precipitation": {"20240501": 2.0}, "temperature": {"20240501": 20.0}}))
        return FakeResponse(payload({"precipitation": values, "temperature": temperatures}))

    monkeypatch.setattr(nasa_power_service, "urlopen", fake_urlopen)
    nasa_power_service.fetch_monthly_weather(49.8, 73.1, 2024, 5)
    nasa_power_service.fetch_monthly_weather(49.8, 73.1, 2024, 5)
    assert len(calls) == 3
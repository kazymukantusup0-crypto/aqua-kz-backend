from datetime import date

from app.schemas import ObservationInput
from app.services.drought_service import calculate_risk, normalize


def make_observation(**overrides: float) -> ObservationInput:
    values = {
        "latitude": 51.1694,
        "longitude": 71.4491,
        "date": date(2026, 6, 1),
        "ndvi": 0.55,
        "ndmi": 0.10,
        "temperature_c": 27.0,
        "precipitation_mm": 9.0,
        "soil_moisture": 0.25,
    }
    values.update(overrides)
    return ObservationInput(**values)


def test_normalize_clips_values_safely() -> None:
    assert normalize(-10.0, 0.0, 10.0) == 0.0
    assert normalize(20.0, 0.0, 10.0) == 1.0
    assert normalize(0.0, 0.0, 10.0, inverse=True) == 1.0


def test_favorable_conditions_are_low_risk() -> None:
    result = calculate_risk(
        make_observation(ndvi=0.78, ndmi=0.36, temperature_c=18.0, precipitation_mm=24.0, soil_moisture=0.44)
    )
    assert result.risk_level == "low"
    assert result.risk_score < 0.40


def test_mixed_conditions_are_medium_risk() -> None:
    result = calculate_risk(make_observation())
    assert result.risk_level == "medium"
    assert 0.40 <= result.risk_score < 0.70


def test_dry_hot_conditions_are_high_risk_with_reasons() -> None:
    result = calculate_risk(
        make_observation(ndvi=0.24, ndmi=-0.18, temperature_c=40.0, precipitation_mm=0.0, soil_moisture=0.07)
    )
    assert result.risk_level == "high"
    assert result.risk_score >= 0.70
    assert "High air temperature" in result.reasons

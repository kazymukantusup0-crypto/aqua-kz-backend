from app.services.recommendation_service import generate_recommendations


def test_september_karaganda_recommendations_are_specific_and_non_predictive() -> None:
    result = generate_recommendations({
        "risk_score": 0.2685,
        "risk_level": "low",
        "trend": "worsening",
        "drought_fraction": 0.4834,
        "precipitation_anomaly": -0.4279,
        "temperature_anomaly": -1.8656,
        "rainfall_stress": 0.4279,
        "heat_stress": 0.0,
        "ndvi": 0.3016,
        "ndvi_baseline": 0.2939,
        "ndvi_anomaly": 0.0077,
        "vegetation_stress": 0.0,
        "soil_moisture_stress": None,
        "data_quality": "partial",
    })
    driver_keys = {driver["key"] for driver in result["drivers"]}
    recommendation_keys = {item["key"] for item in result["recommendations"]}
    assert {"data_quality", "rainfall_deficit", "satellite_drought_pattern"} <= driver_keys
    assert {"field_check", "check_root_zone_moisture", "repeat_analysis_7_days", "monitor_worsening"} <= recommendation_keys
    assert any("43% ниже" in driver["detail"] for driver in result["drivers"])
    assert any("48.3%" in driver["detail"] for driver in result["drivers"])
    assert "агрономическое заключение" in result["disclaimer"]
    assert "vegetation_decline" not in driver_keys
    assert "ирригац" not in " ".join(item["title"] for item in result["recommendations"]).lower()


def test_high_heat_and_vegetation_case_uses_conditional_irrigation_wording() -> None:
    result = generate_recommendations({
        "risk_score": 0.8,
        "risk_level": "high",
        "trend": "worsening",
        "drought_fraction": 0.5,
        "precipitation_anomaly": -0.1,
        "temperature_anomaly": 3.0,
        "heat_stress": 0.6,
        "ndvi_anomaly": -0.1,
        "vegetation_stress": 0.5,
        "soil_moisture_stress": None,
        "data_quality": "partial",
    })
    titles = " ".join(item["title"] for item in result["recommendations"])
    details = " ".join(item["detail"] for item in result["recommendations"])
    assert "irrigation" not in titles.lower()
    assert "При наличии системы орошения" in details
    assert "влажности почвы" in details


def test_low_risk_without_strong_drivers_still_recommends_monitoring() -> None:
    result = generate_recommendations({
        "risk_score": 0.1,
        "risk_level": "low",
        "trend": "stable",
        "drought_fraction": 0.1,
        "data_quality": "segmentation_only",
    })
    assert result["recommendations"][0]["key"] == "routine_monitoring"
    assert result["drivers"][0]["key"] == "data_quality"


def test_null_values_never_crash_or_create_fake_drivers() -> None:
    result = generate_recommendations({
        "risk_score": None,
        "risk_level": None,
        "trend": None,
        "drought_fraction": None,
        "precipitation_anomaly": None,
        "temperature_anomaly": None,
        "heat_stress": None,
        "ndvi_anomaly": None,
        "vegetation_stress": None,
        "soil_moisture_stress": None,
        "data_quality": "partial",
    })
    assert result["drivers"]
    assert result["recommendations"]
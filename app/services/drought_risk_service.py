"""Transparent, missing-data-aware drought risk assessment."""

from __future__ import annotations

from dataclasses import dataclass


WEIGHTS = {
    "vegetation_stress": 0.30,
    "rainfall_stress": 0.25,
    "soil_moisture_stress": 0.20,
    "heat_stress": 0.15,
    "segmentation_stress": 0.10,
}
# When segmentation is available, retain at least one quarter of the partial-data
# score. This prevents a nearly half-patch segmentation signal from becoming a
# low-risk result solely because two unrelated indicators are unavailable.
MIN_SEGMENTATION_SHARE = 0.35


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    risk_level: str
    component_stress: dict[str, float | None]
    missing_inputs: list[str]
    explanation: str
    data_quality: str


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _negative_anomaly(value: float | None, severe: float = 1.0) -> float | None:
    return None if value is None else _clip(-value / severe)


def _positive_anomaly(value: float | None, severe: float = 5.0) -> float | None:
    return None if value is None else _clip(value / severe)


def assess_drought_risk(
    *,
    drought_fraction: float,
    ndvi: float | None = None,
    ndmi: float | None = None,
    precipitation_anomaly: float | None = None,
    precipitation_deficit: float | None = None,
    soil_moisture_anomaly: float | None = None,
    temperature_anomaly: float | None = None,
    vegetation_stress: float | None = None,
    rainfall_stress: float | None = None,
    soil_moisture_stress: float | None = None,
    heat_stress: float | None = None,
    precipitation_mm: float | None = None,
    temperature_c: float | None = None,
) -> RiskResult:
    """Calculate a weighted score using only measured/derived indicators."""
    if vegetation_stress is None:
        vegetation_values = []
        if ndvi is not None:
            vegetation_values.append(_clip((0.8 - ndvi) / 1.8))
        if ndmi is not None:
            vegetation_values.append(_clip((0.5 - ndmi) / 1.5))
        vegetation_stress = sum(vegetation_values) / len(vegetation_values) if vegetation_values else None
    if rainfall_stress is None:
        rainfall_stress = _negative_anomaly(precipitation_anomaly)
        if rainfall_stress is None and precipitation_deficit is not None:
            rainfall_stress = _clip(precipitation_deficit)
    if soil_moisture_stress is None:
        soil_moisture_stress = _negative_anomaly(soil_moisture_anomaly)
    if heat_stress is None:
        heat_stress = _positive_anomaly(temperature_anomaly)

    components = {
        "vegetation_stress": vegetation_stress,
        "rainfall_stress": rainfall_stress,
        "soil_moisture_stress": soil_moisture_stress,
        "heat_stress": heat_stress,
        "segmentation_stress": _clip(drought_fraction),
    }
    available = {name: value for name, value in components.items() if value is not None}
    available_weights = {name: WEIGHTS[name] for name in available}
    denominator = sum(available_weights.values())
    if "segmentation_stress" in available and available_weights["segmentation_stress"] / denominator < MIN_SEGMENTATION_SHARE:
        target = MIN_SEGMENTATION_SHARE * denominator
        adjustment = target - available_weights["segmentation_stress"]
        other_weight_total = denominator - available_weights["segmentation_stress"]
        available_weights["segmentation_stress"] = target
        for name in available_weights:
            if name != "segmentation_stress":
                available_weights[name] -= adjustment * available_weights[name] / other_weight_total
    score = _clip(
        sum(float(value) * available_weights[name] for name, value in available.items()) / sum(available_weights.values())
    )
    missing = [name for name in WEIGHTS if name not in available]
    if len(available) == 1 and "segmentation_stress" in available:
        quality = "segmentation_only"
    elif len(missing) == 0:
        quality = "complete"
    else:
        quality = "partial"
    if score < 0.34:
        level = "low"
    elif score < 0.67:
        level = "medium"
    elif score < 0.85:
        level = "high"
    else:
        level = "critical"
    missing_text = ", ".join(missing) if missing else "none"
    enrichment_text = ""
    if precipitation_anomaly is not None:
        enrichment_text += f" Rainfall for this month was approximately {abs(precipitation_anomaly):.0%} " \
            f"{'below' if precipitation_anomaly < 0 else 'above'} the historical baseline."
    if temperature_anomaly is not None:
        enrichment_text += f" Temperature was about {temperature_anomaly:+.1f} C relative to the historical baseline."
    enrichment_suffix = f"{enrichment_text} " if enrichment_text else ""
    explanation = (
        f"{level.capitalize()} drought risk ({score:.2f}): segmentation detected "
        f"drought-like conditions in {drought_fraction:.1%} of the patch. "
        f"{enrichment_suffix}Data quality is {quality}; unavailable inputs: {missing_text}. Requires field verification."
    )
    return RiskResult(round(score, 4), level, components, missing, explanation, quality)

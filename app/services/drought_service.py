"""Deterministic and explainable baseline drought-risk calculations."""

from __future__ import annotations

from app.config import (
    CONTRIBUTOR_FACTOR_THRESHOLD,
    NDMI_RANGE,
    NDVI_RANGE,
    PRECIPITATION_MM_RANGE,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
    RISK_WEIGHTS,
    SOIL_MOISTURE_RANGE,
    TEMPERATURE_C_RANGE,
)
from app.schemas import ObservationInput, RiskAssessment, RiskLevel


def normalize(value: float, lower: float, upper: float, *, inverse: bool = False) -> float:
    """Safely clip a value into a configured range and normalize it to 0..1."""
    if upper <= lower:
        raise ValueError("Normalization upper bound must exceed lower bound")
    clipped = min(max(value, lower), upper)
    normalized = (clipped - lower) / (upper - lower)
    return 1.0 - normalized if inverse else normalized


def get_risk_level(score: float) -> RiskLevel:
    if score < RISK_LOW_MAX:
        return "low"
    if score < RISK_MEDIUM_MAX:
        return "medium"
    return "high"


def calculate_risk(observation: ObservationInput) -> RiskAssessment:
    """Score drought pressure using documented local-MVP ranges and weights.

    This is an educational rule-based baseline, not a validated drought forecast.
    """
    factors = {
        "ndvi": normalize(observation.ndvi, *NDVI_RANGE, inverse=True),
        "ndmi": normalize(observation.ndmi, *NDMI_RANGE, inverse=True),
        "temperature_c": normalize(observation.temperature_c, *TEMPERATURE_C_RANGE),
        "precipitation_mm": normalize(
            observation.precipitation_mm, *PRECIPITATION_MM_RANGE, inverse=True
        ),
        "soil_moisture": normalize(
            observation.soil_moisture, *SOIL_MOISTURE_RANGE, inverse=True
        ),
    }
    score = sum(factors[name] * RISK_WEIGHTS[name] for name in RISK_WEIGHTS)
    score = round(min(max(score, 0.0), 1.0), 4)

    reason_map = {
        "ndvi": "Low vegetation greenness (NDVI)",
        "ndmi": "Low vegetation water content (NDMI)",
        "temperature_c": "High air temperature",
        "precipitation_mm": "Low recent precipitation",
        "soil_moisture": "Low soil moisture",
    }
    reasons = [
        reason_map[name]
        for name, factor in factors.items()
        if factor >= CONTRIBUTOR_FACTOR_THRESHOLD
    ]
    if not reasons:
        reasons = ["No major drought-pressure indicator exceeded the configured threshold"]

    risk_level = get_risk_level(score)
    explanation = f"{risk_level.capitalize()} drought risk ({score:.2f}): " + "; ".join(reasons)
    return RiskAssessment(
        risk_score=score,
        risk_level=risk_level,
        reasons=reasons,
        explanation=explanation,
    )

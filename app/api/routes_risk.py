"""Risk calculation, zones, and summary routes."""

from fastapi import APIRouter

from app.schemas import ObservationInput, ObservationWithRisk, RiskAssessment, RiskSummary
from app.services.drought_service import calculate_risk
from app.services.satellite_service import load_sample_observations_with_risk

router = APIRouter(prefix="/api", tags=["risk"])


@router.get("/risk-zones", response_model=list[ObservationWithRisk])
def get_risk_zones() -> list[ObservationWithRisk]:
    return load_sample_observations_with_risk()


@router.get("/risk-summary", response_model=RiskSummary)
def get_risk_summary() -> RiskSummary:
    observations = load_sample_observations_with_risk()
    counts = {level: sum(item.risk_level == level for item in observations) for level in ("low", "medium", "high")}
    total = len(observations)
    return RiskSummary(
        total_observations=total,
        low_count=counts["low"],
        medium_count=counts["medium"],
        high_count=counts["high"],
        average_risk_score=round(sum(item.risk_score for item in observations) / total, 4) if total else 0.0,
    )


@router.post("/analyze", response_model=RiskAssessment)
def analyze_observation(observation: ObservationInput) -> RiskAssessment:
    return calculate_risk(observation)

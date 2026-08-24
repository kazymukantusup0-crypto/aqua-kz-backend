"""Routes for locally stored synthetic observations."""

from fastapi import APIRouter

from app.schemas import ObservationWithRisk
from app.services.satellite_service import load_sample_observations_with_risk

router = APIRouter(prefix="/api", tags=["observations"])


@router.get("/observations", response_model=list[ObservationWithRisk])
def get_observations() -> list[ObservationWithRisk]:
    return load_sample_observations_with_risk()

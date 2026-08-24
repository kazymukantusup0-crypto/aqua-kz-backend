"""Future-facing satellite data interface; intentionally offline for this MVP."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

from app.config import SAMPLE_OBSERVATIONS_PATH
from app.schemas import ObservationInput, ObservationRecord, ObservationWithRisk
from app.services.drought_service import calculate_risk


class SatelliteProvider(Protocol):
    """Contract for a future Sentinel Hub or Copernicus provider adapter."""

    def fetch_observation(
        self, latitude: float, longitude: float, observed_on: date
    ) -> ObservationInput:
        """Fetch and derive indicators for a location and date."""


class OfflineSatelliteService:
    """Explicit placeholder that prevents accidental network use in local development."""

    def fetch_observation(
        self, latitude: float, longitude: float, observed_on: date
    ) -> ObservationInput:
        raise NotImplementedError(
            "Satellite integrations are not connected in this local synthetic-data MVP. "
            "Implement a Sentinel Hub/Copernicus adapter that fulfils SatelliteProvider."
        )


def load_sample_observations_with_risk() -> list[ObservationWithRisk]:
    """Load local synthetic observations and attach deterministic risk metadata."""
    data = pd.read_csv(SAMPLE_OBSERVATIONS_PATH)
    required_columns = {
        "id", "latitude", "longitude", "date", "ndvi", "ndmi",
        "temperature_c", "precipitation_mm", "soil_moisture",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Sample observations are missing columns: {sorted(missing_columns)}")

    observations: list[ObservationWithRisk] = []
    for row in data.to_dict(orient="records"):
        observation = ObservationRecord.model_validate(row)
        assessment = calculate_risk(observation)
        observations.append(ObservationWithRisk(**observation.model_dump(), **assessment.model_dump()))
    return observations

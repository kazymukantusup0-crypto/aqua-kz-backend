"""Pydantic contracts for API inputs and outputs."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]


class ObservationInput(BaseModel):
    """Environmental indicators used by baseline and ML risk estimates."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90, description="WGS84 latitude")
    longitude: float = Field(ge=-180, le=180, description="WGS84 longitude")
    date: date
    ndvi: float = Field(ge=-1, le=1)
    ndmi: float = Field(ge=-1, le=1)
    temperature_c: float = Field(ge=-60, le=70)
    precipitation_mm: float = Field(ge=0, le=1000)
    soil_moisture: float = Field(ge=0, le=1)


class ObservationRecord(ObservationInput):
    id: str = Field(min_length=1, max_length=100)


class RiskAssessment(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    reasons: list[str]
    explanation: str


class ObservationWithRisk(ObservationRecord, RiskAssessment):
    pass


class RiskSummary(BaseModel):
    total_observations: int = Field(ge=0)
    low_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    high_count: int = Field(ge=0)
    average_risk_score: float = Field(ge=0, le=1)


class DroneTarget(BaseModel):
    id: str
    latitude: float
    longitude: float
    date: date
    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["high"]
    reason: str
    recommended_action: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    version: str


class TrainingMetrics(BaseModel):
    status: Literal["trained", "not_trained"]
    accuracy: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)
    training_rows: int | None = Field(default=None, ge=1)


class PredictionResponse(BaseModel):
    drought_label: Literal[0, 1]
    prediction: Literal["not_drought_prone", "drought_prone"]
    drought_probability: float = Field(ge=0, le=1)


class DatasetExportRequest(BaseModel):
    """Parameters for a remote Earth Engine patch export job."""

    model_config = ConfigDict(extra="forbid")

    regions: list[Literal["akmola", "karaganda"]] = Field(
        default_factory=lambda: ["akmola", "karaganda"], min_length=1
    )
    year: int = Field(default=2024, ge=2017, le=2100)
    max_patches: int = Field(default=120, ge=1, le=5_000)


class DatasetGenerationResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str
    exported_patches: int | None = Field(default=None, ge=0)


class CoordinateInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    year: int = Field(default=2024, ge=2017, le=2100)


class SegmentationInferenceResponse(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    drought_threshold: float = Field(ge=0, le=1)
    drought_fraction: float = Field(ge=0, le=1)
    mask: list[list[int]]


class PersistedInferenceRun(BaseModel):
    id: str
    created_at: str
    source_type: Literal["upload", "coordinates"]
    upload_filename: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    year: int | None = None
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    drought_threshold: float = Field(ge=0, le=1)
    drought_fraction: float = Field(ge=0, le=1)
    drought_pixel_count: int = Field(ge=0)
    total_pixel_count: int = Field(ge=1)
    mask_url: str
    preview_url: str
    overlay_url: str

"""Pydantic contracts for API inputs and outputs."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]
MonitoringRiskLevel = Literal["low", "medium", "high", "critical"]
Trend = Literal["improving", "stable", "worsening", "insufficient_data"]


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
    month: int | None = Field(default=None, ge=1, le=12)


class SegmentationInferenceResponse(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    drought_threshold: float = Field(ge=0, le=1)
    drought_fraction: float = Field(ge=0, le=1)
    mask: list[list[int]]
    requested_year: int | None = None
    requested_month: int | None = Field(default=None, ge=1, le=12)
    scene_start_date: date | None = None
    scene_end_date: date | None = None
    satellite_source: str | None = None


class PersistedInferenceRun(BaseModel):
    id: str
    created_at: str
    source_type: Literal["upload", "coordinates"]
    upload_filename: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    year: int | None = None
    requested_year: int | None = None
    requested_month: int | None = Field(default=None, ge=1, le=12)
    scene_start_date: date | None = None
    scene_end_date: date | None = None
    satellite_source: str | None = None
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    drought_threshold: float = Field(ge=0, le=1)
    drought_fraction: float = Field(ge=0, le=1)
    drought_pixel_count: int = Field(ge=0)
    total_pixel_count: int = Field(ge=1)
    mask_url: str
    preview_url: str
    overlay_url: str


class MonitoringLocationCreate(BaseModel):
    """A named geographic point monitored over time."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200, description="Human-readable monitoring label")
    latitude: float = Field(ge=-90, le=90, description="WGS84 latitude")
    longitude: float = Field(ge=-180, le=180, description="WGS84 longitude")
    region: str | None = Field(default=None, max_length=200, description="Optional Kazakhstan province or region")


class MonitoringLocationPatch(BaseModel):
    """Fields that may be changed without recreating a monitoring location."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    region: str | None = Field(default=None, max_length=200)
    active: bool | None = None


class MonitoringLocation(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    region: str | None = None
    created_at: str
    active: bool


class MonitoringAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(default=2024, ge=2017, le=2100)
    month: int | None = Field(default=None, ge=1, le=12, description="Observation month; defaults to January when omitted")
    observation_date: date | None = Field(default=None, description="Optional exact observation date")
    enrich: bool = Field(default=False, description="Attempt supported environmental enrichment providers")
    metadata: dict[str, str] | None = None


class BackfillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date
    enrich: bool = False
    force: bool = Field(default=False, description="Reserved for explicit replacement behavior; currently duplicates are skipped")


class RecommendationDriver(BaseModel):
    key: str
    severity: Literal["info", "warning", "danger"]
    title: str
    detail: str


class RecommendationItem(BaseModel):
    key: str
    priority: Literal["low", "medium", "high"]
    title: str
    detail: str


class RecommendationSet(BaseModel):
    drivers: list[RecommendationDriver]
    recommendations: list[RecommendationItem]
    disclaimer: str


class PersistedObservation(BaseModel):
    id: str
    location_id: str
    observed_at: date
    inference_run_id: str
    year: int
    month: int
    drought_fraction: float = Field(ge=0, le=1)
    drought_pixel_count: int = Field(ge=0)
    total_pixel_count: int = Field(ge=1)
    ndvi: float | None = Field(default=None, ge=-1, le=1)
    ndmi: float | None = Field(default=None, ge=-1, le=1)
    ndvi_baseline: float | None = Field(default=None, ge=-1, le=1)
    ndmi_baseline: float | None = Field(default=None, ge=-1, le=1)
    ndvi_anomaly: float | None = Field(default=None, ge=-2, le=2)
    ndmi_anomaly: float | None = Field(default=None, ge=-2, le=2)
    precipitation_mm: float | None = Field(default=None, ge=0, le=1000)
    precipitation_anomaly: float | None = Field(default=None, ge=-10, le=10)
    temperature_c: float | None = Field(default=None, ge=-60, le=70)
    temperature_anomaly: float | None = Field(default=None, ge=-30, le=30)
    soil_moisture: float | None = Field(default=None, ge=0, le=1)
    soil_moisture_anomaly: float | None = Field(default=None, ge=-10, le=10)
    vegetation_stress: float | None = Field(default=None, ge=0, le=1)
    rainfall_stress: float | None = Field(default=None, ge=0, le=1)
    soil_moisture_stress: float | None = Field(default=None, ge=0, le=1)
    heat_stress: float | None = Field(default=None, ge=0, le=1)
    segmentation_stress: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    risk_level: MonitoringRiskLevel
    trend: Trend
    explanation: str
    data_quality: Literal["complete", "partial", "segmentation_only"]
    data_source: str
    metadata: dict[str, str] | None = None
    created_at: str
    recommendations: RecommendationSet | None = None


class MonitoringRunSummary(PersistedInferenceRun):
    pass


class MonitoringObservationResponse(PersistedObservation):
    preview_url: str
    overlay_url: str
    inference_run: MonitoringRunSummary | None = None


class AnalyzeResponse(BaseModel):
    location: MonitoringLocation
    inference_run: MonitoringRunSummary
    observation: MonitoringObservationResponse
    component_stress: dict[str, float | None]
    risk_score: float
    risk_level: MonitoringRiskLevel
    trend: Trend
    explanation: str
    data_quality: Literal["complete", "partial", "segmentation_only"]
    preview_url: str
    overlay_url: str
    recommendations: RecommendationSet | None = None


class TimelineSummary(BaseModel):
    latest_risk_score: float | None = None
    latest_risk_level: MonitoringRiskLevel | None = None
    latest_drought_fraction: float | None = None
    observation_count: int = Field(ge=0)
    trend: Trend
    risk_score_change: float | None = None
    drought_fraction_change: float | None = None
    data_quality_notes: list[str]


class TimelineResponse(BaseModel):
    location: MonitoringLocation
    observations: list[MonitoringObservationResponse]
    summary: TimelineSummary


class LatestResponse(BaseModel):
    location: MonitoringLocation
    observation: MonitoringObservationResponse


class BackfillResponse(BaseModel):
    location_id: str
    created: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    periods: list[str]


class SeasonalOutlookResponse(BaseModel):
    status: Literal["unavailable", "available"]
    forecast_available: bool
    message: str
    limitations: list[str]
    probabilities: dict[str, float] | None = None

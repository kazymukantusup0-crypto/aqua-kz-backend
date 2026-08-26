"""Monitoring locations, explainable observations, and historical timelines."""

from __future__ import annotations

from datetime import date
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas import (
    AnalyzeResponse,
    BackfillRequest,
    BackfillResponse,
    LatestResponse,
    MonitoringAnalyzeRequest,
    MonitoringLocation,
    MonitoringLocationCreate,
    MonitoringLocationPatch,
    MonitoringObservationResponse,
    MonitoringRunSummary,
    SeasonalOutlookResponse,
    TimelineResponse,
    TimelineSummary,
)
from app.services.drought_risk_service import assess_drought_risk
from app.services import inference_run_service, monitoring_service
from app.services.nasa_power_service import fetch_monthly_weather
from app.services.vegetation_service import VegetationEnrichmentError, enrich_patch
from app.services.segmentation_service import SegmentationModelUnavailableError, infer_patch
from ml.export_gee_dataset import (
    EarthEngineConfigurationError,
    SatelliteImageryUnavailableError,
    fetch_patch_for_coordinates,
    scene_date_range,
)


router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


def _location_or_404(location_id: str) -> dict:
    location = monitoring_service.get_location(location_id)
    if location is None:
        raise HTTPException(status_code=404, detail="Monitoring location was not found.")
    return location


def _run_summary(run: dict) -> MonitoringRunSummary:
    return MonitoringRunSummary(
        id=run["id"], created_at=run["created_at"], source_type=run["source_type"],
        upload_filename=run.get("upload_filename"), latitude=run.get("latitude"), longitude=run.get("longitude"),
        year=run.get("year"), width=run["width"], height=run["height"],
        requested_year=run.get("requested_year"), requested_month=run.get("requested_month"),
        scene_start_date=run.get("scene_start_date"), scene_end_date=run.get("scene_end_date"),
        satellite_source=run.get("satellite_source"),
        drought_threshold=run["drought_threshold"], drought_fraction=run["drought_fraction"],
        drought_pixel_count=run["drought_pixel_count"], total_pixel_count=run["total_pixel_count"],
        mask_url=f"/api/ml/runs/{run['id']}/mask", preview_url=f"/api/ml/runs/{run['id']}/preview",
        overlay_url=f"/api/ml/runs/{run['id']}/overlay",
    )


def _observation_response(observation: dict, run: dict | None = None) -> MonitoringObservationResponse:
    response = dict(observation)
    response["preview_url"] = f"/api/ml/runs/{observation['inference_run_id']}/preview"
    response["overlay_url"] = f"/api/ml/runs/{observation['inference_run_id']}/overlay"
    response["inference_run"] = _run_summary(run) if run is not None else None
    return MonitoringObservationResponse.model_validate(response)


def _observation_date(request: MonitoringAnalyzeRequest) -> date:
    if request.observation_date is not None:
        if request.observation_date.year != request.year:
            raise HTTPException(status_code=422, detail="observation_date year must match year.")
        if request.month is not None and request.observation_date.month != request.month:
            raise HTTPException(status_code=422, detail="observation_date month must match month.")
        return request.observation_date
    return date(request.year, request.month or 1, 1)


def _perform_analysis(
    location: dict, observed_at: date, enrich: bool, metadata: dict | None,
    requested_month: int | None = None, force: bool = False,
) -> tuple[dict, dict]:
    if not force and monitoring_service.get_observation_for_period(location["id"], observed_at.year, observed_at.month):
        raise HTTPException(status_code=409, detail="An observation already exists for this location and month.")
    try:
        patch = fetch_patch_for_coordinates(
            location["latitude"], location["longitude"], observed_at.year, requested_month
        )
        probability_mask, drought_fraction = infer_patch(patch)
    except EarthEngineConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SegmentationModelUnavailableError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except SatelliteImageryUnavailableError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    scene_start, scene_end = scene_date_range(observed_at.year, requested_month)
    run = inference_run_service.save_inference_run(
        probability_mask, drought_fraction, 0.5, "coordinates",
        latitude=location["latitude"], longitude=location["longitude"], year=observed_at.year,
        requested_year=observed_at.year, requested_month=requested_month,
        scene_start_date=scene_start, scene_end_date=scene_end,
        satellite_source="Sentinel-2 SR Harmonized + Landsat 8/9 Collection 2 Level-2",
    )
    weather = None
    if enrich:
        try:
            weather = fetch_monthly_weather(location["latitude"], location["longitude"], observed_at.year, observed_at.month)
        except Exception:
            weather = None
    vegetation = None
    if enrich:
        try:
            vegetation = enrich_patch(
                patch, location["latitude"], location["longitude"], observed_at.year, observed_at.month
            )
        except VegetationEnrichmentError:
            vegetation = None
    assessment = assess_drought_risk(
        drought_fraction=drought_fraction,
        ndvi=vegetation.ndvi if vegetation else None,
        ndmi=vegetation.ndmi if vegetation else None,
        ndvi_anomaly=vegetation.ndvi_anomaly if vegetation else None,
        ndvi_baseline=vegetation.ndvi_baseline if vegetation else None,
        vegetation_stress=vegetation.vegetation_stress if vegetation else None,
        precipitation_anomaly=weather.precipitation_anomaly if weather else None,
        temperature_anomaly=weather.temperature_anomaly if weather else None,
        rainfall_stress=weather.rainfall_stress if weather else None,
        heat_stress=weather.heat_stress if weather else None,
    )
    previous = monitoring_service.list_observations(location["id"], limit=1000)
    previous_observations = [
        item for item in previous if item["observed_at"] < observed_at.isoformat()
    ]
    previous_observation = previous_observations[-1] if previous_observations else None
    observation = {
        "id": run["id"], "location_id": location["id"], "observed_at": observed_at.isoformat(),
        "inference_run_id": run["id"], "year": observed_at.year, "month": observed_at.month,
        "drought_fraction": drought_fraction, "drought_pixel_count": run["drought_pixel_count"],
        "total_pixel_count": run["total_pixel_count"], "vegetation_stress": assessment.component_stress["vegetation_stress"],
        "ndvi": vegetation.ndvi if vegetation else None,
        "ndmi": vegetation.ndmi if vegetation else None,
        "ndvi_baseline": vegetation.ndvi_baseline if vegetation else None,
        "ndmi_baseline": vegetation.ndmi_baseline if vegetation else None,
        "ndvi_anomaly": vegetation.ndvi_anomaly if vegetation else None,
        "ndmi_anomaly": vegetation.ndmi_anomaly if vegetation else None,
        "precipitation_mm": weather.precipitation_mm if weather else None,
        "precipitation_anomaly": weather.precipitation_anomaly if weather else None,
        "temperature_c": weather.temperature_c if weather else None,
        "temperature_anomaly": weather.temperature_anomaly if weather else None,
        "rainfall_stress": assessment.component_stress["rainfall_stress"],
        "soil_moisture_stress": assessment.component_stress["soil_moisture_stress"],
        "heat_stress": assessment.component_stress["heat_stress"],
        "segmentation_stress": assessment.component_stress["segmentation_stress"],
        "risk_score": assessment.risk_score, "risk_level": assessment.risk_level,
        "trend": monitoring_service.calculate_trend(
            {"risk_score": assessment.risk_score, "drought_fraction": drought_fraction}, previous_observation
        ),
        "explanation": assessment.explanation, "data_quality": assessment.data_quality,
        "data_source": "; ".join(source for source in (
            "Earth Engine segmentation",
            "satellite NDVI/NDMI" if vegetation else None,
            "NASA POWER daily weather" if weather else None,
        ) if source) or "Earth Engine segmentation",
        "metadata": metadata,
    }
    try:
        if force:
            monitoring_service.delete_observation_for_period(location["id"], observed_at.year, observed_at.month)
        saved_observation = monitoring_service.save_observation(observation)
    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            raise HTTPException(status_code=409, detail="An observation already exists for this location and month.") from error
        raise
    return saved_observation, run


@router.post("/locations", response_model=MonitoringLocation, status_code=status.HTTP_201_CREATED)
def create_monitoring_location(request: MonitoringLocationCreate) -> MonitoringLocation:
    try:
        return monitoring_service.create_location(**request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/locations", response_model=list[MonitoringLocation])
def get_monitoring_locations() -> list[MonitoringLocation]:
    return monitoring_service.list_locations()


@router.get("/locations/{location_id}", response_model=MonitoringLocation)
def get_monitoring_location(location_id: str) -> MonitoringLocation:
    return _location_or_404(location_id)


@router.patch("/locations/{location_id}", response_model=MonitoringLocation)
def patch_monitoring_location(location_id: str, request: MonitoringLocationPatch) -> MonitoringLocation:
    _location_or_404(location_id)
    try:
        result = monitoring_service.update_location(location_id, request.model_dump(exclude_unset=True))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return result


@router.delete("/locations/{location_id}", response_model=MonitoringLocation)
def delete_monitoring_location(location_id: str) -> MonitoringLocation:
    _location_or_404(location_id)
    monitoring_service.deactivate_location(location_id)
    return monitoring_service.get_location(location_id)


@router.post("/locations/{location_id}/analyze", response_model=AnalyzeResponse)
def analyze_location(location_id: str, request: MonitoringAnalyzeRequest) -> AnalyzeResponse:
    location = _location_or_404(location_id)
    observation, run = _perform_analysis(
        location, _observation_date(request), request.enrich, request.metadata, request.month
    )
    compact_observation = _observation_response(observation, run)
    return AnalyzeResponse(
        location=location, inference_run=_run_summary(run), observation=compact_observation,
        component_stress={key: getattr(compact_observation, key) for key in (
            "vegetation_stress", "rainfall_stress", "soil_moisture_stress", "heat_stress", "segmentation_stress"
        )}, risk_score=compact_observation.risk_score, risk_level=compact_observation.risk_level,
        trend=compact_observation.trend, explanation=compact_observation.explanation,
        data_quality=compact_observation.data_quality, preview_url=compact_observation.preview_url,
        overlay_url=compact_observation.overlay_url,
    )


@router.get("/locations/{location_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    location_id: str, start_date: date | None = None, end_date: date | None = None,
    limit: int = Query(default=120, ge=1, le=240), include_runs: bool = False,
) -> TimelineResponse:
    location = _location_or_404(location_id)
    observations = monitoring_service.list_observations(location_id, start_date, end_date, limit)
    responses = []
    for observation in observations:
        run = inference_run_service.get_inference_run(observation["inference_run_id"]) if include_runs else None
        responses.append(_observation_response(observation, run))
    latest = observations[-1] if observations else None
    previous = observations[-2] if len(observations) > 1 else None
    summary = TimelineSummary(
        latest_risk_score=latest["risk_score"] if latest else None,
        latest_risk_level=latest["risk_level"] if latest else None,
        latest_drought_fraction=latest["drought_fraction"] if latest else None,
        observation_count=len(observations), trend=latest["trend"] if latest else "insufficient_data",
        risk_score_change=round(latest["risk_score"] - previous["risk_score"], 4) if latest and previous else None,
        drought_fraction_change=round(latest["drought_fraction"] - previous["drought_fraction"], 4) if latest and previous else None,
        data_quality_notes=sorted({item["data_quality"] for item in observations}) if observations else ["No observations available"],
    )
    return TimelineResponse(location=location, observations=responses, summary=summary)


@router.get("/locations/{location_id}/latest", response_model=LatestResponse)
def get_latest(location_id: str) -> LatestResponse:
    location = _location_or_404(location_id)
    observations = monitoring_service.list_observations(location_id, limit=1_000)
    if not observations:
        raise HTTPException(status_code=404, detail="No observations were found for this monitoring location.")
    observation = observations[-1]
    return LatestResponse(location=location, observation=_observation_response(observation))


def _next_month(current: date) -> date:
    return date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)


@router.post("/locations/{location_id}/backfill", response_model=BackfillResponse)
def backfill_location(location_id: str, request: BackfillRequest) -> BackfillResponse:
    location = _location_or_404(location_id)
    start = date(request.start_date.year, request.start_date.month, 1)
    end = date(request.end_date.year, request.end_date.month, 1)
    if start > end:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date.")
    periods = []
    current = start
    while current <= end and len(periods) < 24:
        periods.append(current.isoformat())
        current = _next_month(current)
    if current <= end:
        raise HTTPException(status_code=422, detail="Backfill range cannot exceed 24 months.")
    created = skipped = failed = 0
    for period in periods:
        observed_at = date.fromisoformat(period)
        if monitoring_service.get_observation_for_period(location_id, observed_at.year, observed_at.month):
            skipped += 1
            continue
        try:
            _perform_analysis(location, observed_at, request.enrich, None, observed_at.month)
            created += 1
        except HTTPException:
            failed += 1
    return BackfillResponse(location_id=location_id, created=created, skipped=skipped, failed=failed, periods=periods)


@router.get("/locations/{location_id}/outlook", response_model=SeasonalOutlookResponse)
def get_outlook(location_id: str) -> SeasonalOutlookResponse:
    _location_or_404(location_id)
    return SeasonalOutlookResponse(
        status="unavailable", forecast_available=False,
        message="This version provides observed monitoring and trend analysis; seasonal outlook requires configured probabilistic forecast data.",
        limitations=["No authenticated probabilistic forecast provider is configured.", "Monitoring trends are not a deterministic future forecast."],
    )
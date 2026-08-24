"""Earth Engine export and seven-band segmentation inference endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.schemas import (
    CoordinateInferenceRequest,
    DatasetExportRequest,
    DatasetGenerationResponse,
    SegmentationInferenceResponse,
)
from app.services.dataset_export_service import create_job, get_job, run_job
from app.services.segmentation_service import (
    SegmentationModelUnavailableError,
    infer_patch,
    load_uploaded_npy,
)
from ml.export_gee_dataset import EarthEngineConfigurationError, fetch_patch_for_coordinates, initialize_earth_engine

router = APIRouter(prefix="/api/ml", tags=["segmentation"])
INFERENCE_THRESHOLD = 0.5


def _response_from_probability(mask, drought_fraction: float) -> SegmentationInferenceResponse:
    binary_mask = (mask >= INFERENCE_THRESHOLD).astype("uint8")
    return SegmentationInferenceResponse(
        width=mask.shape[1],
        height=mask.shape[0],
        drought_threshold=INFERENCE_THRESHOLD,
        drought_fraction=drought_fraction,
        mask=binary_mask.tolist(),
    )


@router.post("/dataset-export", response_model=DatasetGenerationResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_dataset_export(
    request: DatasetExportRequest,
    background_tasks: BackgroundTasks,
) -> DatasetGenerationResponse:
    try:
        initialize_earth_engine()
    except EarthEngineConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    job = create_job(request)
    background_tasks.add_task(run_job, job.job_id, request)
    return job


@router.get("/dataset-export/{job_id}", response_model=DatasetGenerationResponse)
def get_dataset_export_status(job_id: str) -> DatasetGenerationResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset export job was not found.")
    return job


@router.post("/infer/upload", response_model=SegmentationInferenceResponse)
async def infer_uploaded_tile(tile: UploadFile = File(...)) -> SegmentationInferenceResponse:
    if not (tile.filename or "").lower().endswith(".npy"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Upload a seven-channel .npy tile.")
    contents = await tile.read()
    try:
        patch = load_uploaded_npy(contents)
        mask, drought_fraction = infer_patch(patch)
    except SegmentationModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return _response_from_probability(mask, drought_fraction)


@router.post("/infer/coordinates", response_model=SegmentationInferenceResponse)
def infer_coordinates(request: CoordinateInferenceRequest) -> SegmentationInferenceResponse:
    try:
        patch = fetch_patch_for_coordinates(request.latitude, request.longitude, request.year)
        mask, drought_fraction = infer_patch(patch)
    except EarthEngineConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except SegmentationModelUnavailableError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return _response_from_probability(mask, drought_fraction)

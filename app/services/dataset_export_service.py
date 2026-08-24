"""In-memory orchestration for explicit Earth Engine dataset-export jobs."""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

from app.schemas import DatasetExportRequest, DatasetGenerationResponse
from ml.export_gee_dataset import export_dataset

_jobs: dict[str, DatasetGenerationResponse] = {}
_jobs_lock = Lock()


def create_job(request: DatasetExportRequest) -> DatasetGenerationResponse:
    job = DatasetGenerationResponse(
        job_id=str(uuid4()),
        status="queued",
        message="Earth Engine dataset export has been queued.",
    )
    with _jobs_lock:
        _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> DatasetGenerationResponse | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def run_job(job_id: str, request: DatasetExportRequest) -> None:
    """Run synchronous remote export in FastAPI's post-response background task."""
    with _jobs_lock:
        _jobs[job_id] = DatasetGenerationResponse(
            job_id=job_id,
            status="running",
            message="Fetching imagery and exporting .npy patches from Earth Engine.",
        )
    try:
        result = export_dataset(request.regions, request.year, request.max_patches)
        completed = DatasetGenerationResponse(
            job_id=job_id,
            status="completed",
            message="Dataset export completed.",
            exported_patches=result.exported_patches,
        )
    except Exception as error:  # Persist user-visible job failure without leaking a traceback.
        completed = DatasetGenerationResponse(
            job_id=job_id,
            status="failed",
            message=f"Dataset export failed: {error}",
        )
    with _jobs_lock:
        _jobs[job_id] = completed

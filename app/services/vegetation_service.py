"""Satellite vegetation indicators derived from the coordinate-analysis patch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from app.services import inference_run_service
from ml.export_gee_dataset import fetch_patch_for_coordinates


BASELINE_YEARS = 10
BASELINE_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
VEGETATION_CACHE_VERSION = "patch-v1"
NDVI_FULL_STRESS_ANOMALY = -0.20


class VegetationEnrichmentError(Exception):
    """Raised when satellite vegetation indicators cannot be derived."""


@dataclass(frozen=True)
class VegetationIndicators:
    ndvi: float | None
    ndmi: float | None
    ndvi_baseline: float | None
    ndmi_baseline: float | None
    ndvi_anomaly: float | None
    ndmi_anomaly: float | None
    vegetation_stress: float | None


def calculate_patch_indices(patch: np.ndarray) -> tuple[float | None, float | None]:
    """Calculate patch means from normalized Blue..SWIR/NDVI bands.

    The existing Sentinel and Landsat masking functions normalize both sources
    before this patch is downloaded: Sentinel SR is multiplied by 0.0001, while
    Landsat Collection 2 SR uses 0.0000275 and an offset of -0.2. Therefore the
    shared NIR, Red, and SWIR1 bands are directly comparable here.
    """
    array = np.asarray(patch, dtype=float)
    if array.ndim != 3 or array.shape[0] < 5:
        raise VegetationEnrichmentError("The satellite patch does not contain the required spectral bands.")
    nir, red, swir1 = array[3], array[2], array[4]
    ndvi_denominator = nir + red
    ndmi_denominator = nir + swir1
    ndvi_valid = np.isfinite(nir) & np.isfinite(red) & (np.abs(ndvi_denominator) > 1e-6)
    ndmi_valid = np.isfinite(nir) & np.isfinite(swir1) & (np.abs(ndmi_denominator) > 1e-6)
    ndvi_values = (nir - red)[ndvi_valid] / ndvi_denominator[ndvi_valid]
    ndmi_values = (nir - swir1)[ndmi_valid] / ndmi_denominator[ndmi_valid]
    return (
        float(np.mean(ndvi_values)) if ndvi_values.size else None,
        float(np.mean(ndmi_values)) if ndmi_values.size else None,
    )


def vegetation_stress_from_anomaly(ndvi_anomaly: float | None) -> float | None:
    """Map NDVI anomaly 0..-0.20 linearly to vegetation stress 0..1."""
    if ndvi_anomaly is None:
        return None
    return float(np.clip(-ndvi_anomaly / abs(NDVI_FULL_STRESS_ANOMALY), 0.0, 1.0))


def _cache_table() -> None:
    with inference_run_service._connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS vegetation_baseline_cache (
                latitude REAL NOT NULL, longitude REAL NOT NULL, month INTEGER NOT NULL,
                indicator_version TEXT NOT NULL, baseline_year INTEGER NOT NULL,
                ndvi REAL NOT NULL, ndmi REAL NOT NULL, cached_at TEXT NOT NULL,
                PRIMARY KEY (latitude, longitude, month, indicator_version, baseline_year)
            )"""
        )


def _cached_baseline(latitude: float, longitude: float, month: int, years: list[int]) -> dict[int, tuple[float, float]]:
    _cache_table()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=BASELINE_CACHE_TTL_SECONDS)
    with inference_run_service._connect() as connection:
        rows = connection.execute(
            "SELECT baseline_year, ndvi, ndmi FROM vegetation_baseline_cache "
            "WHERE latitude = ? AND longitude = ? AND month = ? AND indicator_version = ? AND cached_at >= ?",
            (round(latitude, 2), round(longitude, 2), month, VEGETATION_CACHE_VERSION, cutoff.isoformat()),
        ).fetchall()
    return {row["baseline_year"]: (row["ndvi"], row["ndmi"]) for row in rows if row["baseline_year"] in years}


def _save_baseline(latitude: float, longitude: float, month: int, values: dict[int, tuple[float, float]]) -> None:
    if not values:
        return
    now = datetime.now(timezone.utc).isoformat()
    with inference_run_service._connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO vegetation_baseline_cache "
            "(latitude, longitude, month, indicator_version, baseline_year, ndvi, ndmi, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(round(latitude, 2), round(longitude, 2), month, VEGETATION_CACHE_VERSION, year, ndvi, ndmi, now)
             for year, (ndvi, ndmi) in values.items()],
        )


def _baseline(latitude: float, longitude: float, year: int, month: int) -> tuple[float | None, float | None]:
    years = list(range(year - BASELINE_YEARS, year))
    cached = _cached_baseline(latitude, longitude, month, years)
    for baseline_year in years:
        if baseline_year in cached:
            continue
        try:
            patch = fetch_patch_for_coordinates(latitude, longitude, baseline_year, month)
            ndvi, ndmi = calculate_patch_indices(patch)
        except Exception as error:
            raise VegetationEnrichmentError("A historical satellite vegetation baseline is unavailable.") from error
        if ndvi is None or ndmi is None:
            raise VegetationEnrichmentError("A historical satellite vegetation baseline has no valid pixels.")
        value = {baseline_year: (ndvi, ndmi)}
        _save_baseline(latitude, longitude, month, value)
        cached.update(value)
    if not cached:
        return None, None
    return (
        sum(value[0] for value in cached.values()) / len(cached),
        sum(value[1] for value in cached.values()) / len(cached),
    )


def enrich_patch(patch: np.ndarray, latitude: float, longitude: float, year: int, month: int) -> VegetationIndicators:
    """Derive current indices and their cached same-month historical baseline."""
    ndvi, ndmi = calculate_patch_indices(patch)
    if ndvi is None or ndmi is None:
        return VegetationIndicators(ndvi, ndmi, None, None, None, None, None)
    ndvi_baseline, ndmi_baseline = _baseline(latitude, longitude, year, month)
    ndvi_anomaly = ndvi - ndvi_baseline if ndvi_baseline is not None else None
    ndmi_anomaly = ndmi - ndmi_baseline if ndmi_baseline is not None else None
    return VegetationIndicators(
        ndvi, ndmi, ndvi_baseline, ndmi_baseline, ndvi_anomaly, ndmi_anomaly,
        vegetation_stress_from_anomaly(ndvi_anomaly),
    )
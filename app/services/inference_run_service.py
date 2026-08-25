"""SQLite storage for persistent drought-segmentation inference runs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from app.config import BACKEND_DIR


DB_PATH = BACKEND_DIR / "data" / "aqua_kz.db"
PREDICTIONS_DIR = BACKEND_DIR / "data" / "predictions"
SourceType = Literal["upload", "coordinates"]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_storage() -> None:
    """Create the persistent inference storage on first use."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK (source_type IN ('upload', 'coordinates')),
                upload_filename TEXT,
                latitude REAL,
                longitude REAL,
                year INTEGER,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                drought_threshold REAL NOT NULL,
                drought_fraction REAL NOT NULL,
                drought_pixel_count INTEGER NOT NULL,
                total_pixel_count INTEGER NOT NULL,
                mask_path TEXT NOT NULL
            )
            """
        )


def save_inference_run(
    probability_mask: np.ndarray,
    drought_fraction: float,
    threshold: float,
    source_type: SourceType,
    upload_filename: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Save a binary drought mask and its compact metadata."""
    initialize_storage()
    run_id = str(uuid4())
    binary_mask = (np.asarray(probability_mask) >= threshold).astype(np.uint8)
    mask_path = PREDICTIONS_DIR / f"{run_id}_mask.npy"
    temporary_path = PREDICTIONS_DIR / f".{run_id}_mask.npy.tmp"
    with temporary_path.open("wb") as temporary_file:
        np.save(temporary_file, binary_mask, allow_pickle=False)
    temporary_path.replace(mask_path)
    metadata = {
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "upload_filename": Path(upload_filename).name if upload_filename else None,
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "width": int(binary_mask.shape[1]),
        "height": int(binary_mask.shape[0]),
        "drought_threshold": threshold,
        "drought_fraction": drought_fraction,
        "drought_pixel_count": int(binary_mask.sum()),
        "total_pixel_count": int(binary_mask.size),
        "mask_path": str(mask_path),
    }
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO inference_runs (
                id, created_at, source_type, upload_filename, latitude, longitude, year,
                width, height, drought_threshold, drought_fraction, drought_pixel_count,
                total_pixel_count, mask_path
            ) VALUES (
                :id, :created_at, :source_type, :upload_filename, :latitude, :longitude, :year,
                :width, :height, :drought_threshold, :drought_fraction, :drought_pixel_count,
                :total_pixel_count, :mask_path
            )
            """,
            metadata,
        )
    return metadata


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_inference_run(run_id: str) -> dict[str, Any] | None:
    initialize_storage()
    with _connect() as connection:
        row = connection.execute("SELECT * FROM inference_runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_dict(row)


def list_inference_runs(limit: int = 50) -> list[dict[str, Any]]:
    initialize_storage()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM inference_runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def load_saved_mask(run: dict[str, Any]) -> np.ndarray:
    """Load a stored binary mask without permitting pickle deserialization."""
    return np.load(run["mask_path"], allow_pickle=False)
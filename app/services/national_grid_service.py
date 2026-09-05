"""Persistence and read models for Kazakhstan's coarse national screening grid."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.services import inference_run_service, monitoring_service
from app.services.kazakhstan_grid import (
    DEFAULT_GRID_VERSION,
    SCREENING_DISCLAIMER,
    SCREENING_DISCLAIMER_KEY,
    configured_grid_cells,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assessment_display(data_quality: str | None, has_observation: bool) -> tuple[str, str]:
    """Return display-only interpretation metadata without changing the observation."""
    if not has_observation:
        return "no_data", "Нет данных"
    if data_quality == "segmentation_only":
        return "preliminary_segmentation_signal", "Уверенность: ограниченная — только спутниковая сегментация"
    if data_quality == "partial":
        return "multi_signal_risk", "Уверенность: частичная — доступны не все компоненты"
    if data_quality == "complete":
        return "multi_signal_risk", "Уверенность: достаточная — доступны несколько компонентов"
    return "multi_signal_risk", "Качество данных не уточнено"


def ensure_grid_cells(grid_version: str = DEFAULT_GRID_VERSION) -> list[dict[str, Any]]:
    """Persist definitions only. It never fetches satellite/weather data."""
    monitoring_service.initialize_monitoring_storage()
    definitions = configured_grid_cells(grid_version)
    with inference_run_service._connect() as connection:
        existing = {
            row[0]
            for row in connection.execute(
                "SELECT cell_id FROM national_grid_cells WHERE grid_version = ?", (grid_version,)
            )
        }
        for cell in definitions:
            if cell["cell_id"] in existing:
                continue
            location_id = str(uuid4())
            created_at = _now()
            connection.execute(
                """INSERT INTO monitoring_locations
                (id, name, latitude, longitude, region, created_at, active, location_type)
                VALUES (?, ?, ?, ?, ?, ?, 1, 'national_grid')""",
                (location_id, f"National screening {cell['cell_id']}", cell["latitude"], cell["longitude"], "Kazakhstan grid", created_at),
            )
            connection.execute(
                """INSERT INTO national_grid_cells
                (grid_version, cell_id, location_id, center_latitude, center_longitude,
                 cell_size_degrees, bounds_json, geometry_json, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (
                    grid_version, cell["cell_id"], location_id, cell["latitude"], cell["longitude"],
                    cell["cell_size_degrees"], json.dumps(cell["bounds"]), json.dumps(cell["geometry"]), created_at,
                ),
            )
    return list_grid_cells(grid_version)


def list_grid_cells(grid_version: str = DEFAULT_GRID_VERSION) -> list[dict[str, Any]]:
    monitoring_service.initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        rows = connection.execute(
            "SELECT * FROM national_grid_cells WHERE grid_version = ? AND active = 1 ORDER BY cell_id", (grid_version,)
        ).fetchall()
    cells = []
    for row in rows:
        cell = dict(row)
        cell["bounds"] = json.loads(cell.pop("bounds_json"))
        cell["geometry"] = json.loads(cell.pop("geometry_json")) if cell.get("geometry_json") else None
        cell["active"] = bool(cell["active"])
        cells.append(cell)
    return cells


def get_grid_cell(grid_version: str, cell_id: str) -> dict[str, Any] | None:
    for cell in list_grid_cells(grid_version):
        if cell["cell_id"] == cell_id:
            return cell
    return None


def record_status(
    grid_version: str,
    cell_id: str,
    year: int,
    month: int,
    status: str,
    failure_reason: str | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    monitoring_service.initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        connection.execute(
            """INSERT INTO national_grid_status
            (grid_version, cell_id, year, month, status, failure_reason, provenance, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(grid_version, cell_id, year, month) DO UPDATE SET
              status = excluded.status, failure_reason = excluded.failure_reason,
              provenance = excluded.provenance, updated_at = excluded.updated_at""",
            (grid_version, cell_id, year, month, status, failure_reason, json.dumps(provenance) if provenance else None, _now()),
        )


def grid_results(
    year: int,
    month: int,
    grid_version: str = DEFAULT_GRID_VERSION,
    risk_level: str | None = None,
    data_quality: str | None = None,
    has_observation: bool | None = None,
) -> list[dict[str, Any]]:
    cells = ensure_grid_cells(grid_version)
    with inference_run_service._connect() as connection:
        rows = connection.execute(
            """SELECT c.*, s.status AS recorded_status, s.failure_reason, s.provenance, s.updated_at,
                      o.*
               FROM national_grid_cells c
               LEFT JOIN national_grid_status s ON s.grid_version = c.grid_version
                   AND s.cell_id = c.cell_id AND s.year = ? AND s.month = ?
               LEFT JOIN monitoring_observations o ON o.location_id = c.location_id
                   AND o.year = ? AND o.month = ?
               WHERE c.grid_version = ? AND c.active = 1
               ORDER BY c.cell_id""",
            (year, month, year, month, grid_version),
        ).fetchall()
    results: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        has_ready_observation = row.get("id") is not None
        current_status = "ready" if has_ready_observation and row.get("recorded_status") not in {"stale"} else row.get("recorded_status") or "missing"
        observation: dict[str, Any] | None = None
        if has_ready_observation:
            observation = {key: row.get(key) for key in (
                "risk_score", "risk_level", "trend", "data_quality", "drought_fraction", "ndvi", "ndmi",
                "ndvi_anomaly", "ndmi_anomaly", "precipitation_anomaly", "temperature_anomaly", "explanation",
                "observed_at", "metadata", "created_at",
            )}
            observation["metadata"] = json.loads(observation["metadata"]) if observation["metadata"] else None
        assessment_mode, confidence_label = _assessment_display(
            observation.get("data_quality") if observation else None, has_ready_observation
        )
        item = {
            "cell_id": row["cell_id"], "grid_version": row["grid_version"],
            "latitude": row["center_latitude"], "longitude": row["center_longitude"],
            "cell_size_degrees": row["cell_size_degrees"], "bounds": json.loads(row["bounds_json"]),
            "geometry": json.loads(row["geometry_json"]) if row.get("geometry_json") else None,
            "status": current_status, "failure_reason": row.get("failure_reason"),
            "provenance": json.loads(row["provenance"]) if row.get("provenance") else None,
            "updated_at": row.get("updated_at"), "is_screening_cell": True,
            "disclaimer_key": SCREENING_DISCLAIMER_KEY, "disclaimer": SCREENING_DISCLAIMER,
            "assessment_mode": assessment_mode, "confidence_label": confidence_label,
            "risk_score": observation.get("risk_score") if observation else None,
            "risk_level": observation.get("risk_level") if observation else None,
            "trend": observation.get("trend") if observation else None,
            "data_quality": observation.get("data_quality") if observation else None,
            "drought_fraction": observation.get("drought_fraction") if observation else None,
            "ndvi": observation.get("ndvi") if observation else None,
            "ndmi": observation.get("ndmi") if observation else None,
            "ndvi_anomaly": observation.get("ndvi_anomaly") if observation else None,
            "ndmi_anomaly": observation.get("ndmi_anomaly") if observation else None,
            "precipitation_anomaly": observation.get("precipitation_anomaly") if observation else None,
            "temperature_anomaly": observation.get("temperature_anomaly") if observation else None,
            "explanation": observation.get("explanation") if observation else None,
            "observed_at": observation.get("observed_at") if observation else None,
            "analysis_metadata": observation.get("metadata") if observation else None,
        }
        if observation:
            from app.services.recommendation_service import generate_recommendations
            item["drivers"] = generate_recommendations(observation)["drivers"]
        else:
            item["drivers"] = []
        if risk_level and item["risk_level"] != risk_level:
            continue
        if data_quality and item["data_quality"] != data_quality:
            continue
        if has_observation is not None and has_ready_observation != has_observation:
            continue
        results.append(item)
    return results


def grid_summary(year: int, month: int, grid_version: str = DEFAULT_GRID_VERSION) -> dict[str, Any]:
    cells = grid_results(year, month, grid_version)
    counts = {status: sum(cell["status"] == status for cell in cells) for status in ("ready", "missing", "failed", "stale")}
    risk_counts = {level: sum(cell["status"] == "ready" and cell["risk_level"] == level for cell in cells) for level in ("low", "medium", "high", "critical")}
    updates = [cell["updated_at"] for cell in cells if cell["updated_at"]]
    return {
        "grid_version": grid_version, "year": year, "month": month, "total_configured_cells": len(cells),
        **counts, "risk_counts": risk_counts, "most_recent_update": max(updates) if updates else None,
    }

"""SQLite persistence and deterministic timeline helpers for monitoring."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

from app.services import inference_run_service


def initialize_monitoring_storage() -> None:
    inference_run_service.initialize_storage()
    with inference_run_service._connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_locations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                region TEXT,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS active_location_coordinates "
            "ON monitoring_locations(latitude, longitude) WHERE active = 1"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_observations (
                id TEXT PRIMARY KEY,
                location_id TEXT NOT NULL REFERENCES monitoring_locations(id),
                observed_at TEXT NOT NULL,
                inference_run_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                drought_fraction REAL NOT NULL,
                drought_pixel_count INTEGER NOT NULL,
                total_pixel_count INTEGER NOT NULL,
                ndvi REAL,
                ndmi REAL,
                ndvi_baseline REAL,
                ndmi_baseline REAL,
                ndvi_anomaly REAL,
                ndmi_anomaly REAL,
                precipitation_mm REAL,
                precipitation_anomaly REAL,
                temperature_c REAL,
                temperature_anomaly REAL,
                soil_moisture REAL,
                soil_moisture_anomaly REAL,
                vegetation_stress REAL,
                rainfall_stress REAL,
                soil_moisture_stress REAL,
                heat_stress REAL,
                segmentation_stress REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                trend TEXT NOT NULL,
                explanation TEXT NOT NULL,
                data_quality TEXT NOT NULL,
                data_source TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(location_id, year, month)
            )
            """
        )
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(monitoring_observations)")}
        for column in ("ndvi_baseline", "ndmi_baseline", "ndvi_anomaly", "ndmi_anomaly"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE monitoring_observations ADD COLUMN {column} REAL")


def _row(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    if "metadata" in result:
        result["metadata"] = json.loads(result["metadata"]) if result["metadata"] else None
    result["active"] = bool(result.get("active")) if "active" in result else result.get("active")
    return result


def create_location(name: str, latitude: float, longitude: float, region: str | None) -> dict[str, Any]:
    initialize_monitoring_storage()
    location = {
        "id": str(uuid4()),
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "region": region,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": 1,
    }
    try:
        with inference_run_service._connect() as connection:
            connection.execute(
                "INSERT INTO monitoring_locations "
                "(id, name, latitude, longitude, region, created_at, active) "
                "VALUES (:id, :name, :latitude, :longitude, :region, :created_at, :active)",
                location,
            )
    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            raise ValueError("An active monitoring location already exists at these coordinates.") from error
        raise
    location["active"] = True
    return location


def get_location(location_id: str) -> dict[str, Any] | None:
    initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        row = connection.execute("SELECT * FROM monitoring_locations WHERE id = ?", (location_id,)).fetchone()
    return _row(row)


def list_locations() -> list[dict[str, Any]]:
    initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        rows = connection.execute("SELECT * FROM monitoring_locations ORDER BY active DESC, created_at DESC").fetchall()
    return [_row(row) for row in rows]


def update_location(location_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
    location = get_location(location_id)
    if location is None:
        return None
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return location
    if values.get("active") is True and not location["active"]:
        values["active"] = 1
    elif "active" in values:
        values["active"] = int(values["active"])
    assignments = ", ".join(f"{key} = :{key}" for key in values)
    values["id"] = location_id
    try:
        with inference_run_service._connect() as connection:
            connection.execute(f"UPDATE monitoring_locations SET {assignments} WHERE id = :id", values)
    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            raise ValueError("An active monitoring location already exists at these coordinates.") from error
        raise
    return get_location(location_id)


def deactivate_location(location_id: str) -> bool:
    with inference_run_service._connect() as connection:
        cursor = connection.execute("UPDATE monitoring_locations SET active = 0 WHERE id = ?", (location_id,))
    return cursor.rowcount > 0


def save_observation(values: dict[str, Any]) -> dict[str, Any]:
    initialize_monitoring_storage()
    observation = dict(values)
    observation.setdefault("id", str(uuid4()))
    observation.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    observation["metadata"] = json.dumps(observation["metadata"]) if observation.get("metadata") else None
    columns = list(observation)
    placeholders = ", ".join(f":{column}" for column in columns)
    with inference_run_service._connect() as connection:
        connection.execute(
            f"INSERT INTO monitoring_observations ({', '.join(columns)}) VALUES ({placeholders})", observation
        )
    observation["metadata"] = json.loads(observation["metadata"]) if observation["metadata"] else None
    return observation


def get_observation(observation_id: str) -> dict[str, Any] | None:
    initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        row = connection.execute("SELECT * FROM monitoring_observations WHERE id = ?", (observation_id,)).fetchone()
    return _row(row)


def get_observation_for_period(location_id: str, year: int, month: int) -> dict[str, Any] | None:
    initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        row = connection.execute(
            "SELECT * FROM monitoring_observations WHERE location_id = ? AND year = ? AND month = ?",
            (location_id, year, month),
        ).fetchone()
    return _row(row)


def delete_observation_for_period(location_id: str, year: int, month: int) -> bool:
    """Delete one explicitly targeted observation before a controlled re-analysis."""
    initialize_monitoring_storage()
    with inference_run_service._connect() as connection:
        cursor = connection.execute(
            "DELETE FROM monitoring_observations WHERE location_id = ? AND year = ? AND month = ?",
            (location_id, year, month),
        )
    return cursor.rowcount > 0


def list_observations(location_id: str, start_date: date | None = None, end_date: date | None = None, limit: int = 120) -> list[dict[str, Any]]:
    initialize_monitoring_storage()
    clauses = ["location_id = ?"]
    parameters: list[Any] = [location_id]
    if start_date:
        clauses.append("observed_at >= ?")
        parameters.append(start_date.isoformat())
    if end_date:
        clauses.append("observed_at <= ?")
        parameters.append(end_date.isoformat())
    parameters.append(limit)
    with inference_run_service._connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM monitoring_observations WHERE {' AND '.join(clauses)} "
            "ORDER BY observed_at ASC, created_at ASC LIMIT ?",
            parameters,
        ).fetchall()
    return [_row(row) for row in rows]


def calculate_trend(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "insufficient_data"
    change = float(current["risk_score"]) - float(previous["risk_score"])
    fraction_change = float(current["drought_fraction"]) - float(previous["drought_fraction"])
    if change >= 0.05 or fraction_change >= 0.05:
        return "worsening"
    if change <= -0.05 or fraction_change <= -0.05:
        return "improving"
    return "stable"
"""Prepare high-risk inspection targets without controlling a physical drone."""

from __future__ import annotations

import csv
import io
from typing import Any, Iterable

from app.schemas import DroneTarget, ObservationWithRisk


RECOMMENDED_ACTION = "Schedule visual drone inspection and verify field conditions."


def build_drone_targets(observations: Iterable[ObservationWithRisk]) -> list[DroneTarget]:
    """Convert high-risk observations into descending-priority inspection targets."""
    targets = [
        DroneTarget(
            id=item.id,
            latitude=item.latitude,
            longitude=item.longitude,
            date=item.date,
            risk_score=item.risk_score,
            risk_level="high",
            reason="; ".join(item.reasons),
            recommended_action=RECOMMENDED_ACTION,
        )
        for item in observations
        if item.risk_level == "high"
    ]
    return sorted(targets, key=lambda target: target.risk_score, reverse=True)


def targets_to_csv_rows(targets: Iterable[DroneTarget]) -> list[dict[str, str | float]]:
    """Return CSV-ready dictionaries for a later export endpoint or UI download."""
    return [target.model_dump(mode="json") for target in targets]


def targets_to_csv_text(targets: Iterable[DroneTarget]) -> str:
    """Return a UTF-8 CSV string without touching the filesystem."""
    rows = targets_to_csv_rows(targets)
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def targets_to_geojson(targets: Iterable[DroneTarget]) -> dict[str, Any]:
    """Return GeoJSON-ready FeatureCollection data for future map/export usage."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [target.longitude, target.latitude]},
                "properties": target.model_dump(mode="json", exclude={"latitude", "longitude"}),
            }
            for target in targets
        ],
    }

"""Configurable, boundary-filtered national screening grid for Kazakhstan.

The grid is intentionally coarse. It is a screening index, not a field or
cadastral boundary.  The bundled boundary is the Natural Earth 1:110m Admin 0
Countries Kazakhstan feature (public domain), simplified for this MVP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_GRID_VERSION = "kz-grid-v1"
DEFAULT_CELL_SIZE_DEGREES = 1.0
# Natural Earth Kazakhstan feature bounds, expanded to whole-degree grid edges.
KAZAKHSTAN_GRID_BOUNDS = {"south": 40.0, "west": 46.0, "north": 56.0, "east": 88.0}
BOUNDARY_SOURCE = "Natural Earth 1:110m Admin 0 Countries (public domain)"
SCREENING_DISCLAIMER_KEY = "national_screening_not_field_diagnosis"
SCREENING_DISCLAIMER = (
    "Карта показывает крупномасштабный скрининг по ячейкам. Она не определяет "
    "состояние конкретного поля и требует детального спутникового и полевого подтверждения."
)


def _boundary_ring() -> list[tuple[float, float]]:
    asset = Path(__file__).resolve().parent.parent / "assets" / "kazakhstan_boundary.geojson"
    payload = json.loads(asset.read_text(encoding="utf-8"))
    return [(float(lon), float(lat)) for lon, lat in payload["geometry"]["coordinates"][0]]


def point_in_kazakhstan(latitude: float, longitude: float) -> bool:
    """Return whether a WGS84 point is inside/on the bundled country polygon."""
    ring = _boundary_ring()
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > latitude) != (y2 > latitude):
            crossing = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < crossing:
                inside = not inside
    return inside


def configured_grid_cells(
    grid_version: str = DEFAULT_GRID_VERSION,
    cell_size_degrees: float = DEFAULT_CELL_SIZE_DEGREES,
) -> list[dict[str, Any]]:
    """Build stable grid definitions in memory; this function never writes data."""
    if cell_size_degrees <= 0:
        raise ValueError("cell_size_degrees must be positive")
    bounds = KAZAKHSTAN_GRID_BOUNDS
    lat_count = int((bounds["north"] - bounds["south"]) / cell_size_degrees)
    lon_count = int((bounds["east"] - bounds["west"]) / cell_size_degrees)
    cells: list[dict[str, Any]] = []
    for lat_index in range(lat_count):
        south = bounds["south"] + lat_index * cell_size_degrees
        north = south + cell_size_degrees
        latitude = round((south + north) / 2, 6)
        for lon_index in range(lon_count):
            west = bounds["west"] + lon_index * cell_size_degrees
            east = west + cell_size_degrees
            longitude = round((west + east) / 2, 6)
            if not point_in_kazakhstan(latitude, longitude):
                continue
            cell_id = f"{grid_version}-{lat_index}-{lon_index}"
            geometry = {
                "type": "Polygon",
                "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
            }
            cells.append({
                "cell_id": cell_id,
                "grid_version": grid_version,
                "latitude": latitude,
                "longitude": longitude,
                "cell_size_degrees": cell_size_degrees,
                "bounds": {"south": south, "west": west, "north": north, "east": east},
                "geometry": geometry,
                "active": True,
            })
    return cells


def grid_plan_summary(grid_version: str = DEFAULT_GRID_VERSION) -> dict[str, Any]:
    cells = configured_grid_cells(grid_version)
    candidate_count = int((KAZAKHSTAN_GRID_BOUNDS["north"] - KAZAKHSTAN_GRID_BOUNDS["south"]) * (KAZAKHSTAN_GRID_BOUNDS["east"] - KAZAKHSTAN_GRID_BOUNDS["west"]))
    return {"grid_version": grid_version, "candidate_count": candidate_count, "inside_kazakhstan_count": len(cells), "bounds": KAZAKHSTAN_GRID_BOUNDS, "cell_size_degrees": DEFAULT_CELL_SIZE_DEGREES}

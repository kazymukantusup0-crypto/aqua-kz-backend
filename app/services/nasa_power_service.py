"""Optional NASA POWER weather enrichment for monitoring observations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.services import inference_run_service


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# A current-month request plus one baseline request are made sequentially, so this
# keeps the normal external-call budget near 14 seconds overall.
NASA_POWER_TIMEOUT_SECONDS = 7
BASELINE_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
BASELINE_YEARS = 10
FILL_VALUE = -999.0

# A -100% precipitation anomaly is full rainfall stress; wetter months add no stress.
RAINFALL_ANOMALY_FULL_STRESS = -1.0
# A +6 C anomaly is full heat stress. These deliberately simple bounds are explainable,
# configurable, and are not a calibrated agricultural threshold.
HEAT_ANOMALY_FULL_STRESS = 6.0


class NasaPowerError(Exception):
    """Raised when NASA POWER cannot provide usable data."""


@dataclass(frozen=True)
class MonthlyWeather:
    precipitation_mm: float | None
    temperature_c: float | None
    baseline_precipitation_mm: float | None
    baseline_temperature_c: float | None
    precipitation_anomaly: float | None
    temperature_anomaly: float | None
    rainfall_stress: float | None
    heat_stress: float | None


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _fetch_daily(latitude: float, longitude: float, start: date, end: date) -> dict[str, dict[str, float]]:
    query = urlencode({
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": f"{longitude:.4f}",
        "latitude": f"{latitude:.4f}",
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    })
    request = Request(f"{NASA_POWER_URL}?{query}", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=NASA_POWER_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise NasaPowerError(f"NASA POWER returned HTTP {response.status}.")
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NasaPowerError("NASA POWER weather data is unavailable.") from error
    try:
        parameters = payload["properties"]["parameter"]
        precipitation = parameters["PRECTOTCORR"]
        temperature = parameters["T2M"]
        dates = set(precipitation) | set(temperature)
        return {
            day: {"precipitation": float(precipitation[day]), "temperature": float(temperature[day])}
            for day in dates
            if day in precipitation and day in temperature
            and float(precipitation[day]) != FILL_VALUE and float(temperature[day]) != FILL_VALUE
        }
    except (KeyError, TypeError, ValueError) as error:
        raise NasaPowerError("NASA POWER returned an unexpected response shape.") from error


def _monthly_values(daily: dict[str, dict[str, float]]) -> tuple[float | None, float | None]:
    if not daily:
        return None, None
    precipitation = [values["precipitation"] for values in daily.values()]
    temperature = [values["temperature"] for values in daily.values()]
    return sum(precipitation), sum(temperature) / len(temperature)


def _cache_table() -> None:
    with inference_run_service._connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS nasa_power_baseline_cache (
                latitude REAL NOT NULL, longitude REAL NOT NULL, month INTEGER NOT NULL,
                baseline_year INTEGER NOT NULL, precipitation_mm REAL NOT NULL,
                temperature_c REAL NOT NULL, cached_at TEXT NOT NULL,
                PRIMARY KEY (latitude, longitude, month, baseline_year)
            )"""
        )


def _get_cached_baseline(latitude: float, longitude: float, month: int, years: list[int]) -> dict[int, tuple[float, float]]:
    _cache_table()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=BASELINE_CACHE_TTL_SECONDS)
    with inference_run_service._connect() as connection:
        rows = connection.execute(
            "SELECT baseline_year, precipitation_mm, temperature_c FROM nasa_power_baseline_cache "
            "WHERE latitude = ? AND longitude = ? AND month = ? AND cached_at >= ?",
            (round(latitude, 2), round(longitude, 2), month, cutoff.isoformat()),
        ).fetchall()
    return {row["baseline_year"]: (row["precipitation_mm"], row["temperature_c"]) for row in rows if row["baseline_year"] in years}


def _save_baseline(latitude: float, longitude: float, month: int, values: dict[int, tuple[float, float]]) -> None:
    if not values:
        return
    now = datetime.now(timezone.utc).isoformat()
    with inference_run_service._connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO nasa_power_baseline_cache "
            "(latitude, longitude, month, baseline_year, precipitation_mm, temperature_c, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(round(latitude, 2), round(longitude, 2), month, year, precipitation, temperature, now)
             for year, (precipitation, temperature) in values.items()],
        )


def _baseline(latitude: float, longitude: float, year: int, month: int) -> tuple[float | None, float | None]:
    years = list(range(year - BASELINE_YEARS, year))
    cached = _get_cached_baseline(latitude, longitude, month, years)
    missing_years = [baseline_year for baseline_year in years if baseline_year not in cached]
    if missing_years:
        # One contiguous same-month window keeps this to one bounded request while
        # allowing the response to be filtered to the ten requested calendar months.
        daily = _fetch_daily(latitude, longitude, date(year - BASELINE_YEARS, month, 1), _month_end(year - 1, month))
        fetched: dict[int, tuple[float, float]] = {}
        for baseline_year in missing_years:
            values = _monthly_values({day: value for day, value in daily.items() if day.startswith(f"{baseline_year}{month:02d}")})
            if values[0] is not None and values[1] is not None:
                fetched[baseline_year] = (values[0], values[1])
        _save_baseline(latitude, longitude, month, fetched)
        cached.update(fetched)
    if not cached:
        return None, None
    return (
        sum(value[0] for value in cached.values()) / len(cached),
        sum(value[1] for value in cached.values()) / len(cached),
    )


def _rainfall_stress(anomaly: float | None) -> float | None:
    return None if anomaly is None else min(max(-anomaly / abs(RAINFALL_ANOMALY_FULL_STRESS), 0.0), 1.0)


def _heat_stress(anomaly: float | None) -> float | None:
    return None if anomaly is None else min(max(anomaly / HEAT_ANOMALY_FULL_STRESS, 0.0), 1.0)


def fetch_monthly_weather(latitude: float, longitude: float, year: int, month: int) -> MonthlyWeather:
    """Fetch a month and an approximate ten-year same-month empirical baseline."""
    current_daily = _fetch_daily(latitude, longitude, date(year, month, 1), _month_end(year, month))
    precipitation, temperature = _monthly_values(current_daily)
    baseline_precipitation, baseline_temperature = _baseline(latitude, longitude, year, month)
    precipitation_anomaly = (
        (precipitation - baseline_precipitation) / baseline_precipitation
        if precipitation is not None and baseline_precipitation is not None and baseline_precipitation > 1e-6
        else None
    )
    temperature_anomaly = (
        temperature - baseline_temperature
        if temperature is not None and baseline_temperature is not None else None
    )
    return MonthlyWeather(
        precipitation, temperature, baseline_precipitation, baseline_temperature,
        precipitation_anomaly, temperature_anomaly,
        _rainfall_stress(precipitation_anomaly), _heat_stress(temperature_anomaly),
    )
"""Recompute the Karaganda demo timeline with current enrichment and risk logic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes_monitoring import _perform_analysis
from app.services import monitoring_service


LOCATION_ID = "e7753663-6559-4549-8b3b-156c29d734b1"
YEAR = 2024


def main() -> None:
    location = monitoring_service.get_location(LOCATION_ID)
    if location is None:
        print(f"location_id={LOCATION_ID} error=location_not_found")
        return
    for month in range(5, 10):
        try:
            observation, _run = _perform_analysis(
                location, date(YEAR, month, 1), True, None, month, force=True
            )
            print(
                f"month={month} drought_fraction={observation['drought_fraction']} "
                f"precipitation_mm={observation['precipitation_mm']} "
                f"precipitation_anomaly={observation['precipitation_anomaly']} "
                f"temperature_c={observation['temperature_c']} "
                f"temperature_anomaly={observation['temperature_anomaly']} "
                f"risk_score={observation['risk_score']} risk_level={observation['risk_level']} "
                f"data_quality={observation['data_quality']}"
            )
        except Exception as error:
            print(f"month={month} status=error error={error}")


if __name__ == "__main__":
    main()